from flask import Flask, request, jsonify, render_template
import os
import tempfile
import uuid
from requests.exceptions import HTTPError
import random
from werkzeug.exceptions import RequestEntityTooLarge
from roboflow_utils import roboflow_infer

app = Flask(__name__)
# Increase maximum file size to 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
# Additional configurations for large file uploads
app.config['MAX_CONTENT_PATH'] = None
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

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

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({'error': 'File too large. Please upload images smaller than 50MB.'}), 413

@app.errorhandler(413)
def handle_413(e):
    return jsonify({'error': 'File too large. Please upload images smaller than 50MB.'}), 413

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
            # Only count trash detections
            num_trash = sum(1 for d in detections if d.get('class', '').lower() == 'trash')
            print(num_trash)
            score = min(num_trash * 4, 250) if num_trash > 0 else 0
            score = max(score - get_buffer(score), 0)
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
    # Get nomination and streak info from form
    nominated = request.form.get('nominated', 'false').lower() == 'true'
    try:
        daily_streak = int(request.form.get('daily_streak', '0'))
    except ValueError:
        daily_streak = 0
    try:
        before_result = roboflow_infer(before)
        after_result = roboflow_infer(after)
        before_trash = sum(1 for d in before_result.get('predictions', []) if d.get('class', '').lower() == 'trash')
        after_trash = sum(1 for d in after_result.get('predictions', []) if d.get('class', '').lower() == 'trash')
        before_score = min(before_trash * 4, 250) if before_trash > 0 else 0
        before_score = max(before_score + get_buffer(before_score), 0)
        after_score = min(after_trash * 4, 250) if after_trash > 0 else 0
        after_score = max(after_score - get_buffer(after_score), 0)
        points_awarded = before_score - after_score
        if points_awarded < 0:
            points_awarded = 0
        percent_cleaned = 0
        if before_score > 0:
            percent_cleaned = ((before_score - after_score) / before_score) * 100
            percent_cleaned = max(0, min(100, percent_cleaned))
        # Apply bonuses
        bonus_message = ""
        if nominated:
            percent_cleaned += 10
            bonus_message += "Nominated bonus applied (+10%). "
        if daily_streak > 0:
            percent_cleaned += daily_streak * 1
            bonus_message += f"Daily streak bonus applied (+{daily_streak}%). "
        percent_cleaned = min(percent_cleaned, 100)
        before_is_trashy = before_score > 12
        before_is_clean = not before_is_trashy
        after_is_trashy = after_score > 12
        after_is_clean = not after_is_trashy
        return jsonify({
            'before_score': before_score,
            'after_score': after_score,
            'points_awarded': points_awarded,
            'percentage_cleaned': percent_cleaned,
            'before_is_trashy': before_is_trashy,
            'before_is_clean': before_is_clean,
            'after_is_trashy': after_is_trashy,
            'after_is_clean': after_is_clean,
            'bonus_message': bonus_message.strip()
        })
    except HTTPError as e:
        return jsonify({'error': f'Roboflow API error: {e.response.text}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
