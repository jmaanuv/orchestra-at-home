import tkinter as tk
import tkinter.messagebox
import pymysql
from PIL import Image, ImageTk
import time
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import sys
import subprocess
from playsound import playsound
from threading import Thread
from playsound import playsound


def play(path):
    """
    Play sound file in a separate thread
    (don't block current thread)
    """
    def play_thread_function():
        playsound(path)
    play_thread = Thread(target=play_thread_function)
    play_thread.start()


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

current_volume = -10.0


def change_volume(change):
    global current_volume
    try:
        if change == 1:
            print('inside voldown')
            if current_volume != -10.5:
                current_volume += (-0.5)
                volume.SetMasterVolumeLevel(current_volume, None)
        if change == 2:
            print('inside volup')
            if current_volume != -0.5:
                current_volume += (0.5)
                volume.SetMasterVolumeLevel(current_volume, None)
        print(f"Volume changed to {current_volume}%")
    except (ValueError, KeyboardInterrupt) as error:
        print("Invalid input.")


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.zeros(468 * 3)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    results = model.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image,
                              results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(240, 240, 240), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image,
                              results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image,
                              results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)


actions = np.array(['start', 'volup', 'voldown'])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('actions.h5')

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


class InstructionsPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        # Create a label for the Instructions page
        label = tk.Label(self, text="Instructions Page")
        label.pack(pady=10)

        # Create buttons to navigate to the other pages
        button1 = tk.Button(self, text="Go to Page 1", command=lambda: controller.show_frame(PageOne))
        button2 = tk.Button(self, text="Go to Page 2", command=lambda: controller.show_frame(PageTwo))
        button3 = tk.Button(self, text="Go to Page 3", command=lambda: controller.show_frame(PageThree))
        button4 = tk.Button(self, text="Go to Page 4", command=lambda: controller.show_frame(PageFour))
        button5 = tk.Button(self, text="Go to Page 5", command=lambda: controller.show_frame(PageFive))
        button6 = tk.Button(self, text="Back", command=lambda: controller.show_frame(AfterLoginPage))
        button6.pack(pady=10)
        button1.pack(pady=10)
        button2.pack(pady=10)
        button3.pack(pady=10)
        button4.pack(pady=10)
        button5.pack(pady=10)


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        # Create a label for page two
        label = tk.Label(self, text="Page One")
        label.pack(pady=10)

        # Create an image and display it on the page
        image = tk.PhotoImage(file="images/volume_up.PNG")
        label_image = tk.Label(self, image=image)
        label_image.image = image
        label_image.pack(pady=10)

        # Create buttons to navigate to the other pages
        button1 = tk.Button(self, text="Go to Page 2", command=lambda: controller.show_frame(PageTwo))
        button2 = tk.Button(self, text="Go back to Instructions Page",
                            command=lambda: controller.show_frame(InstructionsPage))
        button1.pack(pady=10)
        button2.pack(pady=10)


class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        # Create a label for page two
        label = tk.Label(self, text="Page Two")
        label.pack(pady=10)

        # Create an image and display it on the page
        image = tk.PhotoImage(file="images/volume_down.PNG")
        label_image = tk.Label(self, image=image)
        label_image.image = image
        label_image.pack(pady=10)

        # Create buttons to navigate to the other pages
        button1 = tk.Button(self, text="Go to Page 3", command=lambda: controller.show_frame(PageThree))
        button2 = tk.Button(self, text="Go back to Instructions Page",
                            command=lambda: controller.show_frame(InstructionsPage))
        button1.pack(pady=10)
        button2.pack(pady=10)


class PageThree(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        # Create a label for page three
        label = tk.Label(self, text="Page Three")
        label.pack(pady=10)

        # Create an image and display it on the page
        image = tk.PhotoImage(file="images/start.png")
        label_image = tk.Label(self, image=image)
        label_image.image = image
        label_image.pack(pady=10)

        # Create buttons to navigate to the other pages
        button1 = tk.Button(self, text="Go to Page 4", command=lambda: controller.show_frame(PageFour))
        button2 = tk.Button(self, text="Go back to Instructions Page",
                            command=lambda: controller.show_frame(InstructionsPage))
        button1.pack(pady=10)
        button2.pack(pady=10)


class PageFour(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        # Create a label for page three
        label = tk.Label(self, text="Page Four")
        label.pack(pady=10)

        # Create an image and display it on the page
        image = tk.PhotoImage(file="images/pace_fast.PNG")
        label_image = tk.Label(self, image=image)
        label_image.image = image
        label_image.pack(pady=10)

        # Create buttons to navigate to the other pages
        button1 = tk.Button(self, text="Go to Page 5", command=lambda: controller.show_frame(PageFive))
        button2 = tk.Button(self, text="Go back to Instructions Page",
                            command=lambda: controller.show_frame(InstructionsPage))
        button1.pack(pady=10)
        button2.pack(pady=10)


class PageFive(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)

        # Create a label for page three
        label = tk.Label(self, text="Page Five")
        label.pack(pady=10)

        # Create an image and display it on the page
        image = tk.PhotoImage(file="images/pace_slow.PNG")
        label_image = tk.Label(self, image=image)
        label_image.image = image
        label_image.pack(pady=10)

        # Create buttons to navigate to the other pages
        button1 = tk.Button(self, text="Go to Page 4", command=lambda: controller.show_frame(PageFour))
        button3 = tk.Button(self, text="Start playing", command=lambda: controller.show_frame(StartOrchestraPage))
        button2 = tk.Button(self, text="Go back to Instructions Page",command=lambda: controller.show_frame(InstructionsPage))
        button1.pack(pady=10)
        button3.pack(pady=10)
        button2.pack(pady=10)


class RegisterPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Register", font=("Arial", 18))
        label.pack(pady=10, padx=10)

        email_label = tk.Label(self, text="username:")
        email_label.pack(pady=5, padx=10)

        email_entry = tk.Entry(self)
        email_entry.pack(pady=5, padx=10)

        password_label = tk.Label(self, text="Password:")
        password_label.pack(pady=5, padx=10)

        password_entry = tk.Entry(self, show="*")
        password_entry.pack(pady=5, padx=10)

        register_button = tk.Button(self, text="Register",
                                    command=lambda: self.register_user(controller, email_entry.get(),
                                                                       password_entry.get()))
        register_button.pack(pady=10, padx=10)
        back_button = tk.Button(self, text="Back", command=lambda: controller.show_frame(WelcomePage))
        back_button.pack(pady=10, padx=10)

    @staticmethod
    def register_user(controller, email, password):
        if email == '' or password == '' or email != '':
            tkinter.messagebox.showerror('static user', 'use admin for username and password for login')
        # else:
        #     try:
                # conn = pymysql.connect(host='localhost', user='root', password='root')
                # myc = conn.cursor()
            # except:
            #     tkinter.messagebox.showerror('error', 'database connection problem')
            # try:
            #     myc.execute('use users')
            #     query = 'insert into userdata(email, password) values(%s,%s)'
            #     myc.execute(query, (email, password))
            #     conn.commit()
            #     conn.close()
            #     tkinter.messagebox.showinfo('Success', 'Registration successful')
            #     controller.show_frame(LoginPage)
            # except:
            #     tkinter.messagebox.showerror('error', 'user already exists with current user')


class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Login", font=("Arial", 18))
        label.pack(pady=10, padx=10)

        email_label = tk.Label(self, text="user:")
        email_label.pack(pady=5, padx=10)

        email_entry = tk.Entry(self)
        email_entry.pack(pady=5, padx=10)

        password_label = tk.Label(self, text="Password:")
        password_label.pack(pady=5, padx=10)

        password_entry = tk.Entry(self, show="*")
        password_entry.pack(pady=5, padx=10)

        login_button = tk.Button(self, text="Login",
                                 command=lambda: self.login_user(controller, email_entry.get(), password_entry.get()))
        login_button.pack(pady=10, padx=10)

        back_button = tk.Button(self, text="Back", command=lambda: controller.show_frame(WelcomePage))
        back_button.pack(pady=10, padx=10)

    @staticmethod
    def login_user(controller, email, password):
        if email == '' or password == '':
            tkinter.messagebox.showerror('error', 'all fields are required')
        elif email=='admin' and password=='admin':
            tkinter.messagebox.showinfo('success', 'login success')
            controller.show_frame(AfterLoginPage)
        #     try:
        #         conn = pymysql.connect(host='localhost', user='root', password='root')
        #         myc = conn.cursor()
        #     except:
        #         tkinter.messagebox.showerror('error', 'database connection problem')
        #     myc.execute('use users')
        #     myc.execute('select * from userdata where email=%s and password=%s', (email, password))
        #     if myc.fetchone() is not None:
        #         tkinter.messagebox.showinfo('success', 'login success')
        #         controller.show_frame(AfterLoginPage)
        #     else:
        #         tkinter.messagebox.showerror('error', 'invalid credentials')


class WelcomePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Welcome to My App", font=("Arial", 18))
        label.pack(pady=10, padx=10)

        button = tk.Button(self, text="register", command=lambda: controller.show_frame(RegisterPage))
        button.pack(pady=10, padx=10)

        register_button = tk.Button(self, text="Login", command=lambda: controller.show_frame(LoginPage))
        register_button.pack(pady=10, padx=10)


class AfterLoginPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="logged in successfully", font=("Arial", 18))
        label.pack(pady=10, padx=10)

        button = tk.Button(self, text="See Instructions on how to use this app", command=lambda: controller.show_frame(InstructionsPage))
        button.pack(pady=10, padx=10)
        button = tk.Button(self, text="Start the orchestra", command=lambda: controller.show_frame(StartOrchestraPage))
        button.pack(pady=10, padx=10)
        button = tk.Button(self, text="Logout", command=lambda: controller.show_frame(WelcomePage))
        button.pack(pady=10, padx=10)


class StartOrchestraPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Start Orchestra Page", font=("Arial", 18))
        label.pack(pady=10, padx=10)
        self.button = tk.Button(self, text="Start!", width=50, command= self.startOrchestra)
        self.button.pack()


    def startOrchestra(self):
        window_width = 300
        window_height = 500
        x_pos = 1100
        y_pos = (1100 - window_height) // 2
        app.destroy()
        sequence = []
        sentence = []
        threshold = 0.8
        interval = 2
        prev_time = time.time()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
        with mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9,static_image_mode=False, model_complexity=1) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence = actions[np.argmax(res)]
                        else:
                            sentence.append(actions[np.argmax(res)])
                    if len(sentence) > 8:
                        sentence = sentence[-8:]
                    image = prob_viz(res, actions, image, colors)
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                if time.time() - prev_time >= interval:
                    if sentence == 'voldown':
                        change_volume(1)
                    elif sentence == 'volup':
                        change_volume(2)
                    elif sentence == 'start':
                        print('let the show begin!')
                        cv2.destroyAllWindows()
                        play('myMusic.mp3')
                    cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    prev_time = time.time()
                cv2.imshow('Orchestra', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.geometry("800x800+0+0")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (LoginPage, RegisterPage, WelcomePage, AfterLoginPage, StartOrchestraPage, InstructionsPage, PageOne, PageTwo, PageThree, PageFour, PageFive):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(WelcomePage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


app = App()
app.mainloop()
