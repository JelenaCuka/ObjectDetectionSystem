import tkinter as tk  # python 3
from tkinter import font  as tkfont  # python 3
from tkinter import Frame
from typing import List, Tuple
from ObjectDetector import ObjectDetector
from tkinter import filedialog
from Model import Model
class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title_font = tkfont.Font(family='Arial Nova', size=18, weight="bold")  # , slant="italic")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")


        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Wellcome to object detection system", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        f = Frame(self, height=3, width=1000, bg="white")
        f.pack()

        button1 = tk.Button(self, text="Camera Detection", command=lambda: self.openPageCamDetection(), padx=20, pady=10)
        button2 = tk.Button(self, text="Image Detection", command=lambda: controller.show_frame("PageTwo"), padx=20, pady=10)
        button1.pack()
        button2.pack()

        self.model = Model.getInstance()

        f = Frame(self, height=3, width=1000, bg="white")
        f.pack()

        v = tk.IntVar()
        v.set(1)  # initializing the choice, i.e. Python

        trainedModels: List[Tuple[str, int]] = [
            ("ssd_inception_v2_coco_2018_01_28", 1),
            ("faster_rcnn_inception_v2_coco_2018_01_28", 2),
            ("ssd_mobilenet_v2_coco_2018_03_29", 3),
            ("facessd_mobilenet_v2_quantized_320x320_open_image_v4", 4),
            ("intis_Model", 5)
        ]

        def showChoice():
            #print(v.get())
            #model.name = v.get()
            #print(trainedModels[model.name ][0])
            #TODO change the model!!!

            self.model.set_name(trainedModels[v.get()][0])
            #self.model
            print(self.model.get_name())
            if v.get() == 4 :
                self.model.set_bool_custom_trained(True)
            else :
                self.model.set_bool_custom_trained(False)
            #print(self.model.get_bool_custom_trained() )


        #f = Frame(self, height=3, width=1000, bg="white")
        f.pack()

        tk.Label(self,
                 text="""Choose model that will be used
        for detection:""",
                 justify=tk.LEFT,
                 padx=20, pady=5).pack()

        for index, name in enumerate(trainedModels):
            tk.Radiobutton(self,
                           text=name,
                           indicatoron=0,
                           width=20,
                           padx=30,
                           pady=10,
                           variable=v,
                           command=showChoice,
                           value=index).pack(anchor=tk.CENTER)
        f.pack()
        row = tk.Frame(self)
        lab = tk.Label(row, width=22, text="Enter directory"+": ", anchor='w')
        ent = tk.Entry(row)
        ent.insert(0, "")
        row.pack(side=tk.TOP,
                 fill=tk.X,
                 padx=5,
                 pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT,
                 expand=tk.YES,
                 fill=tk.X)
        f.pack()
        print("Enter directory="+ent.get())
       # entries[field] = ent


    def openPageCamDetection(self):
        print("openPageCamDetection")
        self.controller.show_frame("PageOne")
        PageOne(self,self.controller).startDetecting()
        #PageOne.startDetecting()



class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Camera Detection", font=controller.title_font)
        label.pack(side="top", fill="x", padx=30, pady=20)
        f = Frame(self, height=3, width=1000, bg="white")
        f.pack()
        button = tk.Button(self, text="Return to main menu",
                           command=lambda: controller.show_frame("StartPage"), padx=20, pady=10)
        button.pack()
        f = Frame(self, height=3, width=1000, bg="white")
        f.pack()
        self.od = ObjectDetector()
        #TODO RUN CAM DETECTION ->CALL ->OD.DETECTFROMCAM ON INIT ->ON BACK PRESS STOP JE HEHE

    def startDetecting(self):
        print("startDetecting page1 cam")
        self.od.detectOcjectsFromCamera()




class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Image Detection", font=controller.title_font)
        label.pack(side="top", fill="x", padx=30, pady=20)

        f = Frame(self, height=3, width=1000, bg="white")
        f.pack()
        button = tk.Button(self, text="Return to main menu",
                           command=lambda: controller.show_frame("StartPage"), padx=20, pady=10)
        button.pack()
        f = Frame(self, height=3, width=1000, bg="white")
        f.pack()
        # TODO RUN IMAGE DETECTION ->CALL ->OD.DETECTFROMCAM ON INIT ->ON BACK PRESS STOP JE HEHE
        self.od = ObjectDetector()

        #da se poziva kad se otvori!
        #root = tk.Tk()
        #root.withdraw()
        #file_path = filedialog.askopenfilename()
        #print(file_path)

    def startDetecting(self):
        print("startDetecting page1 cam")
        self.od.detectOcjectsFromImages()

if __name__ == "__main__":
    app = SampleApp()

    app.mainloop()

