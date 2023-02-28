
import tkinter as tk
import tkinter.messagebox
from tkinter.filedialog import askopenfilename,askdirectory
import json
import torch
import  os
from PIL import Image, ImageTk, ImageFile
import predict
import model

class Frame(tk.Tk):
    def __init__(self):
        super().__init__()
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.filename = None
        self.fileDir = None
        self.title("水果蔬菜识别")
        self.geometry("1200x700")

        self.index = 0
        self.files = []
        # self.identifier = Identifier()
        self.init()
    def init(self):
        #中间标题
        self.label = tk.Label(self, text = "水果蔬菜识别", font=('Times', 40, 'bold'))
        self.label.place(relx = 0.4)

        self.label1 = tk.Label(self, text = "图片路径", font=('Times', 10, 'bold'))
        self.label1.place(x = 20, y = 150)

        self.label2 = tk.Label(self, text="图片路径", font=('Times', 10, 'bold'))
        self.label2.place(x=20, y=200)
        # 文件路径
        self.e1 = tk.Entry(self)
        self.e1.place(x=100,y=150)
        self.e2 = tk.Entry(self)
        self.e2.place(x=100,y=200)

        # 创建按钮
        self.b1 = tk.Button(self, text="选择图片文件", command=self.getpath)
        self.b1.place(x=250,y=150)

        self.b2 = tk.Button(self, text="选择图片文件夹", command=self.getDirpath)
        self.b2.place(x=250,y=200)
        #显示上传图片
        # self.img_gif = None
        # self.label_img = tk.Label(self, image=self.img_gif)
        self.lableShowImage = tk.Label(self)

        self.lableShowImage.place(relx = 0.5,y = 300)
        #识别按钮
        self.b3 = tk.Button(self, text="单张图片识别",command=self.do_identify)
        self.b3.place(x=450, y=150)
        self.b4 = tk.Button(self, text="批量图片识别", command = self.do_identifyDir)
        self.b4.place(x=450, y=200)

        #识别结果
        self.label3 = tk.Label(self, text="识别结果", font=('Times', 10, 'bold'))
        self.label3.place(x=20, y=250)
        self.result = tk.Entry(self,width = 40)
        self.result.place(x=100, y=250)

        #退出系统
        self.exit = tk.Button(self, text="退出系统",command = self.do_exit)
        self.exit.place(x=450, y=250)

    def getpath(self):
        self.filename = askopenfilename()

        self.e1.delete(0, tk.END)
        self.e1.insert(0, self.filename)

        # img_open = Image.open(self.filename)
        # img_png = ImageTk.PhotoImage(img_open)
        # self.label_img = tk.Label(self, image = img_png)

        img_open = Image.open(self.e1.get())
        img = ImageTk.PhotoImage(img_open.resize((400,300)))
        # img = ImageTk.PhotoImage(img_open)
        self.lableShowImage.config(image=img)
        self.lableShowImage.image = img

    def getDirpath(self):
        self.fileDir = askdirectory()
        self.index = 0
        self.e2.delete(0, tk.END)
        self.e2.insert(0, self.fileDir)
        self.files = os.listdir(self.fileDir)

    def do_identify(self):
        result = predict.detect(self.filename, model,device)
        result = ans_indict[result]
        self.result.delete(0, tk.END)
        self.result.insert(0, result)

    def do_identifyDir(self):
        #根据文件夹获取出所有的图片文件

        try:
            img_open = Image.open(self.fileDir + "\\" + self.files[self.index])
            img = ImageTk.PhotoImage(img_open.resize((400, 300)))
            # img = ImageTk.PhotoImage(img_open)
            self.lableShowImage.config(image=img)
            self.lableShowImage.image = img

            result = predict.detect(self.fileDir + "\\" + self.files[self.index], model, device)
            result = ans_indict[result]
            self.result.delete(0, tk.END)
            self.result.insert(0, result)
        except Exception as e:
            tkinter.messagebox.showwarning(title='识别失败', message='文件无法识别')
        finally:

            self.index += 1
            self.index %= len(self.files)

    def do_exit(self):
        self.destroy()





if __name__ == '__main__':
    json_path = './name.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, encoding='utf-8')
    ans_indict = json.load(json_file)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.resnet34(num_classes=25).to(device)

    # load model weights
    weights_path = "resnet34-pre.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    win = Frame()
    win.mainloop()