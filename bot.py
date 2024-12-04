# Bot Imports
import os
import requests
import math
import telebot
from telebot import types
from dotenv import load_dotenv

# Model Imports
import sys
import os
import subprocess
import shutil

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load environment variables from .env file. Your BOT_TOKEN should be there.
load_dotenv()

# Model Preparation
USERS_PATH = 'users'
VIDEO_DIR_NAME = 'videos'
VIDEO_VEC_DIR_NAME = 'video_vectors'
COLLECTION_NAME = "vector_collection"

def create_user(user_name, collection = COLLECTION_NAME):
    if user_name not in os.listdir(USERS_PATH):
        os.mkdir('/'.join([USERS_PATH, user_name]))
    if VIDEO_DIR_NAME not in os.listdir('/'.join([USERS_PATH, user_name])):
        os.mkdir('/'.join([USERS_PATH, user_name, VIDEO_DIR_NAME]))
    if VIDEO_VEC_DIR_NAME not in os.listdir('/'.join([USERS_PATH, user_name])):
        os.mkdir('/'.join([USERS_PATH, user_name, VIDEO_VEC_DIR_NAME]))
    client = QdrantClient(path=f'{USERS_PATH}/{user_name}/db')
    client.recreate_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
    )
    return client

def create_raw_vector_file(input_file, output_file):
    command = [
                "conda/bin/python3.7",
                "video_embedding/video2vec.py",
                "--graph_file", "./Models/inception-v3_image_classify_graph_def.pb",
                "--fcnn_model", "./Models/weibo_MCN_14k_frames30_sfps1.ckpt-done",
                "--input_file", input_file,
                "--output_file", output_file
            ]
    subprocess.run(command)

def process_raw_vector(raw_vector):
    vector = raw_vector.split(',')[-1].split('_')
    vector = np.array([float(x) for x in vector])
    return vector

def add_one_vector_to_bd(processed_vector,
                         video_name,
                         client,
                         collection = COLLECTION_NAME):
    amount_of_vectors = client.count(
    collection_name=collection, 
    exact=True,
    ).count
    client.upsert(
        collection_name=collection,
        points=models.Batch(
            ids=[amount_of_vectors],
            vectors=[processed_vector],
            payloads=[{'source': video_name}]
        )
    )

def calc_vector(video_name, 
                user_name,
                client, 
                collection = COLLECTION_NAME,
                users_path = USERS_PATH, 
                video_dir_name = VIDEO_DIR_NAME, 
                video_vec_dir_name = VIDEO_VEC_DIR_NAME):
    input_file = '/'.join(['.', users_path, user_name, video_dir_name, video_name])
    output_file = '/'.join(['.', users_path, user_name, video_vec_dir_name, '.'.join(video_name.split('.')[:-1])])
    create_raw_vector_file(input_file, output_file)
    processed_vector = process_raw_vector(open(output_file, 'r').readline())
    add_one_vector_to_bd(processed_vector, video_name, client, collection)

def calc_vector_and_get_closest(video_name, 
                                user_name,
                                client, 
                                collection = COLLECTION_NAME,
                                users_path = USERS_PATH, 
                                video_dir_name = VIDEO_DIR_NAME, 
                                video_vec_dir_name = VIDEO_VEC_DIR_NAME,
                                limit = 5):
    input_file = '/'.join(['.', users_path, user_name, video_dir_name, video_name])
    output_file = '/'.join(['.', users_path, user_name, video_vec_dir_name, '.'.join(video_name.split('.')[:-1])])
    create_raw_vector_file(input_file, output_file)
    processed_vector = process_raw_vector(open(output_file, 'r').readline())
    return client.search(
    collection_name=collection,
    query_vector=processed_vector,
    limit=limit
    )

BOT_TOKEN = os.environ.get('BOT_TOKEN')

bot = telebot.TeleBot(BOT_TOKEN)

welcome_prompt = '''Welcome to Database Video Search Bot!'''

all_users = {}

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    """Give a welcome prompt."""
    bot.reply_to(message, welcome_prompt)
    global all_users
    all_users[str(message.chat.id)] = create_user(str(message.chat.id))

@bot.message_handler(commands=['process_videos'])
def process_all_videos(message):
    global all_users
    user_name = str(message.chat.id)
    for video_name in os.listdir('/'.join([USERS_PATH, user_name, VIDEO_DIR_NAME])):
        calc_vector(video_name,
                    user_name,
                    all_users[user_name])
    bot.send_message(message.chat.id, 'Done')

@bot.message_handler(func=lambda message: message.chat.type=='private', content_types=['video']) 
def photo_worker(message): 
    global all_users
    caption = message.caption
    video_name = message.video.file_name
    user_name = str(message.chat.id)
    if message.video.file_size > 20000000:
        bot.send_message(message.chat.id, 'Video too big')
        return None

    file_info = bot.get_file(message.video.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = '/'.join(['.', USERS_PATH, user_name, VIDEO_DIR_NAME, video_name])
    with open(src, "wb") as new_file:
        new_file.write(downloaded_file)
    if caption.lower()=='add_db':
        calc_vector(video_name,
                    user_name,
                    all_users[user_name])
        bot.send_message(message.chat.id, 'Video added to db')
    elif caption.lower()=='get_closest':
        search_result = calc_vector_and_get_closest(video_name,
                                    user_name,
                                    all_users[user_name])
        search_results_message = ''
        search_results_message+='Best match:\n'+str(search_result[0].payload['source'])+ '\t' +str(search_result[0].score)+'\n'
        search_results_message+='Closest matches: '+'\n'
        for result in search_result[1:]:
            search_results_message+=str(result.payload['source']) + '\tScore: ' + str(result.score) +'\n'
        bot.send_message(message.chat.id, search_results_message)
    else:
        bot.send_message(message.chat.id, 'Wrong Command')
        return None
    
bot.infinity_polling()