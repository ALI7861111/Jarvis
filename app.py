from flask import Flask, request, jsonify

app = Flask(__name__)

# In-memory store for items
store = {}
counter = 1


@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "REST API running", "status": "ok"})


@app.route('/items', methods=['GET'])
def get_items():
    # Return all items
    return jsonify(store)


@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    if item_id in store:
        return jsonify({"id": item_id, "value": store[item_id]})
    return jsonify({"error": "not found"}), 404


@app.route('/items', methods=['POST'])
def create_item():
    global counter
    data = request.get_json(force=True) or {}
    if 'value' not in data:
        return jsonify({"error": "missing value"}), 400
    item_id = counter
    counter += 1
    store[item_id] = data['value']
    return jsonify({"id": item_id, "value": data['value']}), 201


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
