#import ImageTk
import inception_v4
from gtts import gTTS 
 
from tkinter import Label, PhotoImage
from tkinter import *
from tkinter import filedialog

from PIL import Image, ImageTk, ImageSequence
import win32com.client 

seeee='choose image first ' 

background='C:/Users/abdel/Desktop/giphy (1).gif'
imgl='C:/Users/abdel/Downloads/source.gif'
imgr='C:/Users/abdel/Downloads/animation-clipart-elephant-841073-8559255.gif'


class Application(tkinter.Frame):
     
    
    x=0
    sav = 'please entter an image'
    def __init__(self, master=None):
        super().__init__(master)
        self.say(z='let’s start')
        self.master = master
        self.pack() 
        self.gif_bg()
        self.create_widgets()
        
        self.panel1.pack(side='left', fill='both', expand='yes')




        
    def animate(self, counter):
            self.panel1.itemconfig(self.image,image=self.sequence[counter])
            self.master.after(80,lambda: self.animate((counter+1) % len(self.sequence))) 

    def gif_bg(self):
            self.panel1 =tkinter.Canvas(self.master, width=600, height=540)
            self.panel1.pack()
            self.sequence =[ImageTk.PhotoImage(self.img)
                                for self.img in ImageSequence.Iterator(
                                        Image.open(
                                                background))]
            self.image =self.panel1.create_image((270), (270), image= self.sequence[0])
            self.animate(1)

    def RtoL(self):
        for j in range(600):
            self.rrr =tkinter.Canvas(self.panel1, width=60, height=50)
            self.rrr.pack()
            self.sequenceR =[ImageTk.PhotoImage(self.img)
                                for self.img in ImageSequence.Iterator(
                                        Image.open(
                                                imgr))]
            self.image =self.rrr.create_image((600-j), (500), image= self.sequenceR[0])
            self.animate(1)
            j=j+4
    def LtoR(self):
        for i in range(600):
            self.LLL =tkinter.Canvas(self.panel1, width=60, height=50)
            self.LLL.pack()
            self.sequenceL =[ImageTk.PhotoImage(self.img)
                                for self.img in ImageSequence.Iterator(
                                        Image.open(
                                                imgl))]
            self.image =self.LLL.create_image((i), (500), image= self.sequenceL[0])
            self.animate(1)
            i=i+4

    def create_widgets(self):
        
        
        ############################################
        ## leftpart#################################
    
        self.leftt = Label(self.panel1)
        self.leftt.pack(side='left')
        
        
        #img_b1 = ImageTk.PhotoImage(Image.open('C:/Users/abdel/Desktop/math -project/codes/INCEPTION V4/hhh/Inception-v4-master/button_open-image.png'))
        self.brwse_button = tkinter.Button(self.leftt, width=10, height=3)
        self.brwse_button["text"] = "Browse"
        self.brwse_button["command"] = self.browse
        self.brwse_button.pack(side='top')
        
        
        self.pred_button = tkinter.Button(self.leftt, width=10, height=3)
        self.pred_button["text"] = "Predict"
        self.pred_button["command"] = self.pred
        self.pred_button.pack(side='top')        
        
        
        self.say_button = tkinter.Button(self.leftt, text="Say again",
                              command=self.say(z=self.sav), width=10, height=3)
        self.say_button.pack(side='top')
        

        self.quit = tkinter.Button(self.leftt, text="QUIT", fg="red",
                              command=self.master.destroy, width=10, height=3)
        self.quit.pack(side='top')
       
        
        ##################################################
        ###right part#############################################
               
        self.rightt = Label(self.panel1)
        self.rightt.pack(side='left')
        
        self.pimg = Label(self.rightt)
        self.pimg.pack(side='top')
        
        self.pathlabel_2 = Label(self.rightt, bd=1, relief="solid", font= "Times 15" ,height =1)
        
        self.pathlabel_1 = Label(self.rightt, bd=1, relief="solid", font= "Times 15" ,height =2)






                              

    
    def browse(self):
         self.filename = filedialog.askopenfilename()
         self.showImg(self.filename)
         
         self.pathlabel_1.config(text= "please click predict")
         self.pathlabel_2.config(text= "                          ")
         self.showop()
         self.cdd=0

         
    def pred(self):
        if self.cdd==0:
             self.pathlabel_1.config(text= "Loading ....       ")
             self.showop()
             self.showop()

             
             self.c ,self.cer =inception_v4.get_pred(self.filename)  
             
             
             self.cer = 'certainity is : '+self.cer
             self.c ='class is : '+self.c
             #spl=self.c.split(",", 1)
             self.sav=self.c
             self.pathlabel_1.config(text=self.c) 
             
             self.pathlabel_2.config(text=self.cer)
             self.showop()
             


             self.cdd =1
            
             self.say(z=self.sav)             
    def showImg(self, f):
        
    #        self.rightt.pack_forget()
            load = Image.open(f)
            load.thumbnail((400,250))
            render = ImageTk.PhotoImage(load)
            
            # labels can be text or images                  
            
            self.pimg.configure(image=render)
            self.pimg.image = render
    
    def showop(self):
        
        self.pathlabel_2.config(bg='black', fg='yellow')             
        
        self.pathlabel_1.config(bg='black', fg='yellow')             
        
        self.leftt.pack(side='left')
        self.brwse_button.pack(side='top')

        self.pimg.pack(side='top')
        self.pathlabel_2.pack( fill= BOTH, side='bottom' )
        self.pathlabel_1.pack( fill= BOTH, side='bottom' )
        
        self.rightt.pack( side='left')
        self.panel1.pack(side='left', fill='both', expand='yes')
        self.pack()

        
    def say(self , z):
        speaker = win32com.client.Dispatch("SAPI.SpVoice") 
        speaker.Speak(z)
                
     
        
        
        
        
        
        
    def set_bg(self,bkg):
        image1 = tkinter.PhotoImage(file=bkg)
        w = image1.width()
        h = image1.height()
        root.geometry("%dx%d+0+0" % (w, h))
        # tk.Frame has no image argument
        # so use a label as a panel/frame
        self.panel1 = tkinter.Label(root, image=image1)
        self.panel1.pack(side='left', fill='both', expand='yes')        
        
        #save the panel's image from 'garbage collection
        self.panel1.image = image1
        

       
            

        
        
        
        
        
if __name__ == '__main__':  


     

    root = tkinter.Tk()
    root.title('Irrationals') 
    root.geometry("540x540")
    root.resizable(False, False)
    
    

    
    app = Application(master=root)
    app.mainloop()

    