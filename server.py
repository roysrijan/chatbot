from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room


# --- Flask App ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Executive Management ---
executives = []
executive_counter = 0


# --- WebSocket Events ---

# When a user wants to start a chat
@socketio.on('join')
def handle_join(data):
    user_type = data.get('type')  # 'user' or 'executive'
    room = data.get('room')       # unique room id for user, or 'executives' for execs
    user_id = data.get('user_id')  # Unique identifier for the user
    join_room(room)
    print(f"{user_id} joined room: {room}")
    if user_type == 'user':
        emit('connect_to_executive', {'room': room, 'type': user_type, 'user_id': user_id, 'to': executives[0]}, room=room)
    if executives:
        executives.pop(0)  # Remove the first executive from the list to connect with the user

# When a customer executive connects, they join the 'executives' room
@socketio.on('executive_online')
def handle_executive_online(data):
    global executive_counter
    # executive_counter += 1
    executive_id = data.get('user_id') #f"executive_{executive_counter}"
    executives.append(executive_id)
    join_room('executives')
    emit('executive_ready', {'msg': f'Executive {executive_id} is online', 'executive_id': executive_id}, room='executives')
    print(f"{executive_id} joined. Executives online: {executives}")

# Relay chat messages between user and executive
@socketio.on('chat_message')
def handle_chat_message(data):
    room = data.get('room')
    message = data.get('message')
    _from = data.get('from')  # 'user' or 'executive'
    _to= data.get('to')  # 'user' or 'executive'
    print(f"Received message in room {room} from {_from}: {message}")
    emit('chat_message', {'from': _from, 'to': _to, 'message': message}, room=room)


if __name__ == '__main__':
    # In a production deployment, the hosting platform will typically manage this
    # but this is useful for local testing.


    socketio.run(app, host='0.0.0.0', port=8080) # Use 0.0.0.0 to be accessible within the container
