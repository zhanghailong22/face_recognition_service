#!/usr/bin/python
# -*- coding: utf-8 -*-

from flask import Flask, Response, request, jsonify
import redis
import face_recognition
import numpy as np


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

pool = redis.ConnectionPool(host='127.0.0.1', port=6379)

# 首页
@app.route('/', methods=['GET'])
def index():
    return '''
    <!doctype html>
    <title>人脸服务</title>
    <a href="upload">人脸录入</a><br>
    <a href="search">人脸搜索</a>
    '''

# 人脸录入页
@app.route('/upload', methods=['GET'])
def uploadHtml():
    return '''
    <!doctype html>
    <title>人脸录入</title>
    <h1>人脸录入</h1>
    <form method="POST" action="upload" enctype="multipart/form-data">
      姓名：<input type="text" name="name"><br>
      <input type="file" name="file" >
      <input type="submit" value="提交">
    </form>
    '''

# 人脸录入
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'code': 500, 'msg': '没有文件'})
    file = request.files['file']
    name = request.form['name']
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) != 1:
        return jsonify({'code': 500, 'msg': '人脸数量有误'})
    face_encodings = face_recognition.face_encodings(image, face_locations)
    # 连数据库
    r = redis.Redis(connection_pool=pool)
    # 录入人名-对应特征向量
    r.set(name, face_encodings[0].tobytes())
    return jsonify({'code': 0, 'msg': '录入成功'})

# 人脸搜索页
@app.route('/search', methods=['GET'])
def searchHtml():
    return '''
    <!doctype html>
    <title>人脸搜索</title>
    <h1>人脸搜索</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="提交">
    </form>
    '''

# 人脸搜索
@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'code': 500, 'msg': '没有文件'})
    file = request.files['file']
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) != 1:
        return jsonify({'code': 500, 'msg': '人脸数量有误'})
    face_encodings = face_recognition.face_encodings(image, face_locations)
    # 连数据库
    r = redis.Redis(connection_pool=pool)
    # 取出所有的人名和它对应的特征向量
    names = r.keys()
    faces = r.mget(names)
    # 组成矩阵，计算相似度（欧式距离）
    matches = face_recognition.compare_faces([np.frombuffer(x) for x in faces], face_encodings[0])
    return jsonify({'code': 0, 'names': [str(name, 'utf-8') for name, match in zip(names, matches) if match]})

if __name__ == '__main__':

    app.run()