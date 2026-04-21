from flask import Flask, request, jsonify, render_template
from src.tag_tree import TagTree
from src.user_profile import UserProfile
from src.recommender import CourseRecommender
import pandas as pd
import os

app = Flask(__name__)

# 初始化组件
model="Qwen/Qwen3-30B-A3B-Instruct-2507"
openai_key="sk-wgholxssmljlxivhonnjivryzmoxzfbfxunpswfmncuaydlx"
base_url="https://api.siliconflow.cn/v1"

tag_tree = TagTree(openai_key)
user_profiler = UserProfile(tag_tree)
courses_df = pd.read_csv('data/tag_course_mapping.csv')
recommender = CourseRecommender(courses_df, user_profiler)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/profile', methods=['POST'])
def get_profile():
    user_data = request.json
    profile = user_profiler.build_profile(user_data)
    return jsonify(profile)

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    user_data = request.json
    profile = user_profiler.build_profile(user_data)
    recommendations = recommender.recommend(profile)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True,port=8000)