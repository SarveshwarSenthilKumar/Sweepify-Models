from flask import Flask, request, jsonify, render_template
import os
import tempfile
import uuid
from requests.exceptions import HTTPError
import random
from roboflow_utils import roboflow_infer

app = Flask(__name__)

def get_buffer(score):
    if score <= 20:
        return 12
    elif score <= 40:
        return 14
    elif score <= 60:
        return 16
    elif score <= 80:
        return 18
    else:
        return 20

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image = request.files['image']
    try:
        try:
            result = roboflow_infer(image)
            detections = result.get('predictions', [])
            num_trash = sum(1 for d in detections if d.get('class', '').lower() == 'trash')
            score = min(num_trash * 20, 100) if num_trash > 0 else 0
            if score > 0:
                buffer = get_buffer(score)
                score = max(score - buffer, 0)
            is_trashy = score > 12
            is_clean = not is_trashy
            return jsonify({
                'score': score,
                'is_trashy': is_trashy,
                'is_clean': is_clean
            })
        except HTTPError as e:
            return jsonify({'error': f'Roboflow API error: {e.response.text}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rate_cleaning', methods=['POST'])
def rate_cleaning():
    if 'before' not in request.files or 'after' not in request.files:
        return jsonify({'error': 'Both before and after images required'}), 400
    before = request.files['before']
    after = request.files['after']
    try:
        before_result = roboflow_infer(before)
        after_result = roboflow_infer(after)
        before_trash = sum(1 for d in before_result.get('predictions', []) if d.get('class', '').lower() == 'trash')
        after_trash = sum(1 for d in after_result.get('predictions', []) if d.get('class', '').lower() == 'trash')
        before_score = min(before_trash * 20, 100) if before_trash > 0 else 0
        after_score = min(after_trash * 20, 100) if after_trash > 0 else 0
        if before_score > 0:
            before_score = min(before_score + get_buffer(before_score), 100)
        if after_score > 0:
            after_score = max(after_score - get_buffer(after_score), 0)
        points_awarded = before_score - after_score
        if points_awarded < 0:
            points_awarded = 0
        return jsonify({
            'before_score': before_score,
            'after_score': after_score,
            'points_awarded': points_awarded
        })
    except HTTPError as e:
        return jsonify({'error': f'Roboflow API error: {e.response.text}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 