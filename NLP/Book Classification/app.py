'''
Author: xiaoyao jiang
LastEditors: xiaoyao jiang
Date: 2020-08-31 14:18:26
LastEditTime: 2020-08-31 14:59:14
FilePath: /newBookClassification/app.py
Desciption:  
'''
from flask import Flask, request
import json
from model import BookClassifier

# 初始化模型， 避免在函数内部初始化，耗时过长
bc = BookClassifier()
bc.load()

app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def gen_ans():
    '''
    @description: 以RESTful的方式获取模型结果, 传入参数为title: 图书标题， desc: 图书描述
    @param {type}
    @return: json格式， 其中包含标签和对应概率
    '''
    result = {}
    title = request.form['title']
    desc = request.form['desc']
    label = bc.predict(title, desc)
    result = {
        "label": label
    }
    return json.dumps(result, ensure_ascii=False)


# python3 -m flask run
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)