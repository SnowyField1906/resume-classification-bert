from flask import Flask, request, jsonify

from model.main import load, train

app = Flask(__name__)

@app.route('/process', methods=['GET'])
def process():
    content = request.args.get('content')

    if content:
        res = load(content)
        return jsonify(res), 201
    else:
        return jsonify({'success': False}), 400
    
@app.route('/train', methods=['POST'])
def retrain():
    loss, acc = train()
    return jsonify({'loss': str(round(loss * 100, 2)), 'acc': str(round(acc * 100, 2))}), 201
    
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify("pong"), 200

if __name__ == '__main__':
    app.run(debug=False)