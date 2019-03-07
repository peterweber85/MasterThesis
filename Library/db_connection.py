
import pymongo
import os
import csv
import pandas as pd


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


def write_labels_from_csv_to_db(db_collection, folder_name, csv_filename):
    """
    Writes both the multi label and the binary label to the db, it takes the info from the csv file
    and matches on the metadata filename.

    :param db_collection: mongodb connection and define collection
    :param folder_name: str
    :param csv_filename: str
    :return:
    """
    with open(folder_name + csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                label_multi_name = row[0]
                label_binary_name = row[1]
                print("loading labels " + label_multi_name + " and " + label_binary_name + " from file " + csv_filename + " to db ...")
                line_count += 1
            else:
                query = {"filename": row[2]}
                new_multi_label = {"$set": {label_multi_name: row[0]}}
                new_binary_label = {"$set": {label_binary_name: row[1]}}
                db_collection.update_one(query, new_multi_label)
                db_collection.update_one(query, new_binary_label)
                line_count += 1
        print(str(line_count-1) + ' labels added to db!')
    return


def delete_all_metadata_with_labels(db_collection, label_name):
    for label in [0,1,2,3,4]:
        query = {label_name: str(label)}
        db_collection.delete_many(query)
    return


def query_filenames_of_labelled_images(db_collection, label_name):
    """
    Returns dataframe with filenames and label of labelled images
    :param db_collection: mongodb collection object
    :param label_name: str
    :return:
    """
    output = list(db_collection.find({label_name: {"$exists": True}}))
    filenames = [(el['filename']) for el in output]
    labels = [(el[label_name]) for el in output]

    return pd.DataFrame({
        'filename': filenames,
        'label': labels
    })