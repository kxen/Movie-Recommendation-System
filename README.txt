Instructions regarding the project setup, configuration and execution.

	The solution was built in an OS X environment (version 10.14.1 (18B75)) using Python 2.7.10.
	The latest version of Apache Spark (version 2.3.2) has been used for the development of the Movie Recommendation 	 Engine. Python's microframework Flask (1.0.2) was used to support the RESTful API on top of the spark Engine.
	For the support of asynchronous Flask calculations, Celery task queue (4.2.1) has been used. The Celery client in 	  Flask communicates to the Celery worker through Redis message broker.

	The following Python modules are required for the successful execution:
		-pyspark.mllib.recommendation
		- shutil
		- os
		- logging
		- math
		- sys
		- pandas
		- flask
		- pyspark.sql
		- json
		- celery
		- calendar
		- time

	The project repos consists of:
		- fishEngine.py
		- fishRecommenderApp.py
		- directory datasets (includes ratings.csv and movies.csv)

	To execute the Movie Recommendation App we have to (execution in order of appearance)
		- start spark master (./start-master.sh)
		- start spark client (./start-slave.sh spark://$(hostname -s):7077)
		- start redis server (redis-server)
		- check redis client (redis-client ping)
		- submit spark job (spark-submit --class org.apache.spark.examples.sparkPi --master spark://$(hostname -		  s):7077  fishRecommenderApp.py)
		- start celery (celery -A fishRecommenderApp.celery worker -l debug)

Details of the implemented recommendation system (e.g., how the modules work and cooperate/communicate, where and how the data is being stored and processed, etc.)

	 The implemented movie recommendation system consists of the following major modules:
		- fishEngine.py
		- fishRecommenderApp.py
	fishEngine.py implements class recomSparkEngine which is responsible for all major functionalities of the Movie 	Recommendation System. During class initialization the following actions are performed:
		- ratings and movie datasets are loaded to RDD spark structures
		- ratings dataset is subsampled and trained with the aid of the ALS algorithm in order to locate the optimal  			dimension of the latent variable of the corresponding matrix factorization model. Out of sample validation 		     is used to identify the optimal rank of the model.
		- the optimal model is persisted.
	Due to lack of time, the ratings and movies files are persisted as csv files. This is the major reason for selecting  	      the small movie-lens dataset. Time permitted I would like to use a cassandra scheme to support movie ratings and     	   appended data.
	Class method "trainALSModel" is responsible for finding the optimal latent variables (the dimension found during class 	       initialization is utilized) that define movies user ratings. The rest of methods are in one-to-one correspondence 	 with the constructed API calls. Of special interest is the "addRating" method which appends user ratings for specific 	       movies in the existing RDD structure and triggers a retraining of the Matrix Factorization model. For each new rating,  	       a separate ratings file ("ratingsAppended.csv") and matrix factorization model ("modelAppended") are created and 	persisted. Ideally, the retraining of the model should occur after adding a sufficient number of ratings. Here, to 	   illustrate the execution of model retraining as a background process, retraining is performed after appending only one 	  rating.

	fishRecommenderApp.py is used to deploy the fishEngine Movie Recommender System as a standalone RESTful API. For the 	     execution of model retraining as a background process, Celery task queue is deployed. In particular, a Celery client 	  is triggered every time a new rating is added to the system and a new model retraining task is required. The Celery 		client communicates with the Celery worker through Redis message broker to retrain the model as a background process.

Description of the API calls
	The following API calls are implemented:
		Get movie info (title, year and genre(s))
			http://127.0.0.1:5000/movieInfo/<movie_id (type:int)>/

		Get top-100 movies (based on average rating and the number of users)
			http://127.0.0.1:5000/topMovies/<top_count (type:int)>

		Get top-10 recommendations for some user (consider only user id as a parameter)
			http://127.0.0.1:5000/user/<user_id (type:int)>/topMovies/<top_count (type:int)>

		Add/remove a user (action=1 -> add user, action=0 -> remove user)
			http://127.0.0.1:5000/user/<user_id (type:int)>/modify/<action (type:int)>

		Mark movie as viewed or rate a movie for a user
			http://127.0.0.1:5000/addRating/user/<user_id (type:int)>/movie/<:movie_id (type:int)>/rating/<rating 			      (type:float)>

		Get all viewed/rated movies of a user
			http://127.0.0.1:5000/getAllMovies/user/<user_id (type:int)>



