
import pymongo
import os


def connect(credentials, db_name):

    # load credentials
    try:
        db_credentials = os.environ[credentials]
        print("DB Credentials from environ")
    except:
        with open(credentials) as f:
            db_credentials = f.readline().strip("\n").strip()
            print("DB Credentials from file")
            f.close

    # connect to db
    try:
        db_conn = pymongo.MongoClient(db_credentials)
        print ("DB connected successfully!!!")
    except pymongo.errors.ConnectionFailure as e:
        print ("Could not connect to DB: %s" % e)

    db = db_conn[db_name]

    return db
