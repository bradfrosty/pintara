from flask import Flask
app = Flask(__name__)

@app.route("/paintings")
def createNewPainting():
    return "Create a new painting"

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
