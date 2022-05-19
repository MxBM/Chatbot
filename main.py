import subprocess
import time


if __name__ == "__main__":
    # creating threads
    network = "cd chatbot & python app.py"
    server = "npm start"
    sts = subprocess.Popen(network, shell=True)
    sts2 = subprocess.Popen(server, shell=True)
    time.sleep(10)
    Call_URL = "http://localhost:3000"
    mycmd = r'start chrome /new-tab {}'.format(Call_URL)
    sts3 = subprocess.Popen(mycmd, shell=True)
