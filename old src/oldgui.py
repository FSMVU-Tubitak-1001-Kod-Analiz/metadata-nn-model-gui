import tkinter as tk
from RecordVoice import record_voice
from SpeechRecognition import recognize_speech
from TextToSpeech import text_to_speech
from TextProcessing import read_text
import threading
import time

def animate_recording():
    global angle
    global start_time
    elapsed_time = time.time() - start_time
    if elapsed_time < 9:
        x0 = 15
        y0 = 15
        x1 = 50
        y1 = 50
        recording_canvas.create_arc(x0, y0, x1, y1, start=angle, width=5, extent=10, style=tk.ARC)
        angle += 2.6
        if angle >= 360:
            angle = 0
        root.after(50, animate_recording)
    else:
        recording_canvas.delete("all")
        recording_canvas.create_line(15, 30, 30, 45, fill="green", width=5)
        recording_canvas.create_line(30, 45, 45, 15, fill="green", width=5)
        
def animate_processing():
    global angle
    global processing_complete
    elapsed_time = time.time() - start_time
    if processing_complete == False:
        processing_canvas.delete("all")
        x0 = 15
        y0 = 15
        x1 = 50
        y1 = 50
        processing_canvas.create_arc(x0, y0, x1, y1, start=angle, width=5, extent=40, style=tk.ARC)
        angle += 20
        if angle >= 360:
            angle = 0
        root.after(50, animate_processing)
    else:
        processing_canvas.delete("all")
        processing_canvas.create_line(15, 30, 30, 45, fill="green", width=5)
        processing_canvas.create_line(30, 45, 45, 15, fill="green", width=5)

def on_record():
    recording_canvas.delete("all")
    processing_canvas.delete("all")
    processing_label.config(text="")
    
    main_thread = threading.Thread(target=main_process)
    main_thread.start()
    record_button.config(state=tk.DISABLED)
    
    global start_time
    global angle
    start_time = time.time()
    angle=0
    animate_recording()
    
    
def main_process():
    recording_label.config(text="Recording...")
    record_voice()
    recording_label.config(text="Recording Complete!")
    
    global processing_complete
    processing_complete = False
    
    global angle
    angle=0
    animate_processing()
    processing_label.config(text="Processing...")
    
    recognize_speech()
    
    file = open('recognizedText.txt', 'r')
    recognized = file.read()
    file.close()
    
    read_text(recognized)
    
    file = open('output.txt', 'r')
    data = file.read()
    file.close()
    
    recognized_label.config(text=recognized)
    result_label.config(text=data)
    processing_complete = True
    processing_label.config(text="Processing Complete!")
    
    if data is '':
        text_to_speech("I couldn't recognize your voice clearly. What I heard was:  " + recognized + ". Can you say it again?")
    else:
        text_to_speech("the answer is "+data)
    record_button.config(state=tk.ACTIVE)
    

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Vocal Calculator")
    root.geometry("500x560")
    root.config(background="gray")
    
    record_button = tk.Button(root, text="Start Recording", font=('Ariel',20), command=on_record)
    record_button.pack(padx=20,pady=20)
    
    help_label = tk.Label(root,background="gray",font=('Ariel',8),text= "To calculate interest: interest for principal ... rate ... duration ...", width=100, height=1)
    help_label.pack(padx=5,pady=5)

    grid = tk.Frame(root,borderwidth=2, relief="solid")
    grid.columnconfigure(0, weight=1)
    grid.columnconfigure(1, weight=1)
    
    recording_canvas = tk.Canvas(grid, width=60, height=60)
    angle = 0
    recording_canvas.grid(row=0,column=0, sticky=tk.E)
    
    processing_canvas = tk.Canvas(grid, width=60, height=60)
    angle = 0
    processing_canvas.grid(row=1,column=0, sticky=tk.E)
    
    recording_label = tk.Label(grid,text= "", width=20, height=5)
    recording_label.grid(row=0,column=1, sticky=tk.W)
    
    processing_label = tk.Label(grid,text="", width=20, height=5)
    processing_label.grid(row=1, column=1, sticky=tk.W)
    
    grid.pack(fill='x')

    decorative_label_1 = tk.Label(root, text="Recognized Text:",background="gray")
    decorative_label_1.pack(padx=5,pady=10)

    recognized_label = tk.Label(root, text="", font=('Ariel',15),width=60,height=2,wraplength=450)
    recognized_label.pack(padx=20,pady=20)

    decorative_label_2 = tk.Label(root, text="Result Calculation:",background="gray")
    decorative_label_2.pack(padx=5,pady=10)


    result_label = tk.Label(root, text="", font=('Ariel',15),width=60)
    result_label.pack(padx=20,pady=20)

    root.mainloop()