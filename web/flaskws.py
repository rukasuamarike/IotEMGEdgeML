from flask import Flask, request
from simple_websocket import Server, ConnectionClosed

app = Flask(__name__)

@app.route("/g", websocket=True)
def g():
    ws = Server(request.environ)
    print("WebSocket connection established")
    try:
        while True:
            data = ws.receive()
            print(f"Received: {data}")
            ws.send(data + "aaa")
    except ConnectionClosed:
        print("WebSocket connection closed")
    return ""

if __name__ == "__main__":
    app.run(port=5000)
