from pyspark.mllib.recommendation import ALS
import shutil
import os
import logging
import math
import calendar
import time
import pandas as pd
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class recomSparkEngine():
    def __init__(self, spark_context, dataset_path, ratings_file, movies_file, model_name, find_best_rank):
        self.sc = spark_context
        self.modelName = model_name
        logger.info("Loading datasets related to Movie recommendations")
        # load datasets related to recommendations (ratings and movies)
        ratings_file = os.path.join(dataset_path, ratings_file)
        logger.info("Ratings file %s", ratings_file)
        ratings_raw = self.sc.textFile(ratings_file)
        ratings_header = ratings_raw.first()
        ratings_raw = ratings_raw.filter(lambda row: row != ratings_header)
        self.ratings_RDD = ratings_raw.map(lambda line: line.split(",")).map(
            lambda columns: (int(columns[0]), int(columns[1]), float(columns[2]))).cache()

        movies_file = os.path.join(dataset_path, movies_file)
        movies_raw = self.sc.textFile(movies_file)
        movies_header = movies_raw.first()
        movies_raw = movies_raw.filter(lambda row: row != movies_header)
        self.movies_RDD = movies_raw.map(lambda line: line.split(",")).map(
            lambda columns: (int(columns[0]), columns[1], columns[2])).cache()
        self.moviesCount=self.movies_RDD.count()
        logger.info("Locating ALS recommendation model parameters")
        # build best rank recommendation model
        # set matrix factorization parameters
        self.seed = 5
        self.iterations = 10
        self.regularization_parameter = 0.1
        self.tolerance = 0.02
        self.best_rank = 12
        if find_best_rank:
            ranks = [4, 8, 12]
            min_error = float('inf')
            self.best_rank = 12
            # Bernouli sampling
            for rank in ranks:
                training_rdd, validation_rdd, test_rdd = self.ratings_RDD.sample(False, 0.01).randomSplit([6, 2, 2], seed=0)
                # estimate best rank of matrix factorization latent variables
                # for rank in ranks:
                model = ALS.train(training_rdd, rank, seed=self.seed, iterations=self.iterations, lambda_= self.regularization_parameter)
                predictions = model.predictAll(validation_rdd.map(lambda x: (x[0], x[1]))).map(lambda r: ((r[0], r[1]), r[2]))
                ratings_and_predictions = validation_rdd.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
                error = math.sqrt(ratings_and_predictions.map(lambda r: (r[1][0] - r[1][1])**2).mean())
                if error < min_error:
                    min_error = error
                    self.best_rank = rank

        # logger.info("Best rank found=%s",self.best_rank)
        # train model
        self.trainALSModel()


    def trainALSModel(self):
        logger.info("Training ALS recommendation model")
        training_rdd, test_rdd = self.ratings_RDD.randomSplit([7, 3], seed=0)
        self.bestRankModel = ALS.train(training_rdd, rank=self.best_rank, seed=self.seed, iterations=self.iterations,
                                       lambda_=self.regularization_parameter)
        test_predictions = self.bestRankModel.predictAll(test_rdd.map(lambda x: (x[0], x[1]))).map(
            lambda r: ((r[0], r[1]), r[2]))
        ratings_and_predictions = test_rdd.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(test_predictions)
        error = math.sqrt(ratings_and_predictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
        logger.info("Best rank model out of sample error %f", error)
        # persist model
        if os.path.isdir(self.modelName):
            try:
                shutil.rmtree(self.modelName)
            except IOError:
                pass
        try:
            self.bestRankModel.save(self.sc, self.modelName)
        except IOError:
            logger.info("Cannot not save matrix factorization model")

    # Get movie info (title, year and genre(s))
    def getMovieInfo(self, movieID):
        return self.movies_RDD.filter(lambda movie: movie[0] == movieID).map(lambda movie: (movie[1], movie[2])).collect()

    # Get top-100 movies (based on average rating and the number of users)
    def getTopMovies(self, topCount):
        # for each movie get its average rating and the number of users who have rated it
        topMovies = self.ratings_RDD.map(lambda x: (x[1], x[2])).aggregateByKey((0, 0), lambda acc, value: (acc[0] + value, acc[1] + 1), lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])).mapValues(
            lambda v: [1.0 * v[0] / v[1], v[1]])
        # sort movies
        sortTopMoviesInfo=topMovies.join(self.movies_RDD).map(lambda x: (x[1][0][0], x[1][0][1], x[1][1])).sortBy(lambda x: (x[1], x[0]), False).take(topCount)
        try:
            assert(topCount == len(sortTopMoviesInfo) and topCount <= self.moviesCount)
        except AssertionError:
            logger.Exception("Returning wrong number of top movies")
        return sortTopMoviesInfo

    # Get top-10 recommendations for some user (consider only user id as a parameter)
    def getUserTopRecommendations(self, user_id, top_count):
        # locate all movies rated by user_id
        ratedMovies = self.ratings_RDD.filter(lambda x: x[0] == user_id).map(lambda x: x[1]).collect()
        # locate unrated movies of user_id
        notRatedMovies_RDD = self.movies_RDD.filter(lambda movie: movie[0] not in ratedMovies).map(lambda x: (user_id, x[0]))
        # use the model to predict ratings for unrated movies
        recommendations_RDD = self.bestRankModel.predictAll(notRatedMovies_RDD)
        movies = self.movies_RDD.map(lambda x: (x[0], (x[1])))
        recoms = recommendations_RDD.map(lambda x: (x[1], (x[2])))
        topRecommendations = movies.join(recoms).map(lambda x: (x[1][0], x[1][1])).sortBy(lambda x: x[1], ascending=False).take(top_count)
        try:
            assert(top_count == len(topRecommendations) and top_count <= self.moviesCount)
        except AssertionError:
            logger.Exception("Returning wrong number of top movies viewed or rated by user")
        return topRecommendations

    # Add (action=1) or remove (action=0) a user
    def modifyUser(self, user_id, action):
        if action == 1: # add user
            # check whether the user already exists
            if len(self.ratings_RDD.filter(lambda user: user[0] == user_id).take(1)) == 0:
                new_user = self.sc.parallelize([(user_id, 0, 0)])
                self.ratings_RDD = self.sc.union([self.ratings_RDD, new_user])
                # dump RDD
                ratings_appended = """%s,%s,%s,%s\n""" % (str(user_id), str(0), str(0), str(calendar.timegm(time.gmtime())))
                with open(os.path.join("ml-small", "ratings.csv"), 'a') as f:
                    f.write(ratings_appended)

        else: # remove user
            self.ratings_RDD = self.ratings_RDD.filter(lambda x: x[0] != user_id)
            ratings_df = pd.read_csv(os.path.join("ml-small", "ratings.csv"))
            try:
                os.remove(os.path.join("ml-small", "ratings.csv"))
            except IOError:
                logger.info("Cannot delete ratings.csv file")
            ratings_df[ratings_df.userId != user_id].to_csv(os.path.join("ml-small", "ratings.csv"), index=False)

    # Mark movie as viewed or rate a movie for a user
    def addRating(self, user_id, movie_id, rating):
        if len(self.ratings_RDD.filter(lambda user: user[0] == user_id and user[1] == movie_id and user[2] == rating).collect()) == 1:
            return False
        logger.info("rating is %s"%self.ratings_RDD.filter(lambda user:user[0]==user_id and user[1] == movie_id).map(lambda x:x[2]).collect())

        new_rating = self.sc.parallelize([(user_id, movie_id, rating)])
        # overwrite existing rating for selected (user, movie) pair in RDD
        filtered_ratings_RDD=self.ratings_RDD.filter(lambda user: user[0] == user_id)
        filtered_ratings_RDD=filtered_ratings_RDD.filter(lambda user: user[1] != movie_id)

        self.ratings_RDD = self.sc.union([self.ratings_RDD.filter(lambda user:user[0] != user_id), filtered_ratings_RDD, new_rating])
        # dump RDD
        ratings_appended = """%s,%s,%s,%s\n""" % (str(user_id), str(movie_id), str(rating), str(calendar.timegm(time.gmtime())))
        with open(os.path.join("ml-small", "ratings.csv"), 'a') as f:
            f.write(ratings_appended)
        return True

    # Get all viewed/rated movies of a user
    def getAllMoviesForUser(self, user_id):
        movies_of_user = self.ratings_RDD.filter(lambda user: user[0] == user_id).map(lambda movie: (movie[1], (movie[2])))
        movies = movies_of_user.join(self.movies_RDD).map(lambda movie: movie[1][1]).collect()
        return movies
