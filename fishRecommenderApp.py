from pyspark.mllib.recommendation import MatrixFactorizationModel
import sys
import os
import shutil
from flask import Flask
from pyspark.sql import SparkSession
import json
import logging
from celery import Celery
from fishEngine import recomSparkEngine

sys.path.append('./python')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

spark = SparkSession.builder.appName("Movie Recommendation Spark Engine").getOrCreate()


@celery.task(bind=True)
def rerun_modeling(self):
    recommendation_engine_append = recomSparkEngine(spark.sparkContext, "ml-small", "ratings.csv", "movies.csv", "modelAppended", False)
    if os.path.isdir("modelAppended"):
        try:
            shutil.rmtree("modelAppended")
        except IOError:
            pass
    recommendation_engine_append.bestRankModel.save(spark.sparkContext, "modelAppended")
    return 1


@app.route("/movieInfo/<int:movie_id>/", methods=["GET"])
def movie_info(movie_id):
    logger.debug("Information for movie %s", movie_id)
    return "%s" % json.dumps(recommendation_engine.getMovieInfo(movie_id))


@app.route("/topMovies/<int:top_count>", methods=["GET"])
def top_movies(top_count):
    logger.debug("# Get top-%s movies (based on average rating and the number of users)", top_count)
    return "%s" % json.dumps(recommendation_engine.getTopMovies(top_count))


@app.route("/user/<int:user_id>/topMovies/<int:top_count>", methods=["GET"])
def get_user_top_recommendations(user_id, top_count):
    logger.debug("# Get top-%s movies for user %s", (top_count, user_id))
    return "%s" % json.dumps(recommendation_engine.getUserTopRecommendations(user_id, top_count))


@app.route("/user/<int:user_id>/modify/<int:action>", methods=["GET"])
def modify_user(user_id, action):
    recommendation_engine.modifyUser(user_id, action)
    if action == 1:
        logger.debug("# Add user %s", user_id)
        return "User %s added" % user_id
    else:
        logger.debug("# Remove user %s", user_id)
        return "User %s removed" % user_id


@app.route("/addRating/user/<int:user_id>/movie/<int:movie_id>/rating/<float:rating>", methods=["GET"])
def add_rating(user_id, movie_id, rating):
    if not recommendation_engine.addRating(user_id, movie_id, rating):
        return "Rating for user %s and movie %s already exists" % (user_id, movie_id)

    result = rerun_modeling.delay()
    result.wait()
    recommendation_engine.bestRankModel = MatrixFactorizationModel.load(spark.sparkContext, "modelAppended")
    return "Rating has been added"


@app.route("/getAllMovies/user/<int:user_id>", methods=["GET"])
def get_user_all_movies(user_id):
    logger.debug("Get all movies for user %s", user_id)
    return "%s" % json.dumps(recommendation_engine.getAllMoviesForUser(user_id))


def create_app(spark_context, dataset_path, ratings_file, movies_file):
    global recommendation_engine

    recommendation_engine = recomSparkEngine(spark_context, dataset_path, ratings_file, movies_file, "model", True)


if __name__ == "__main__":
    create_app(spark.sparkContext, "ml-small", "ratings.csv", "movies.csv")
    app.run()
