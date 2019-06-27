
import tkinter
from dotenv import load_dotenv
import os
load_dotenv("../.env")


def create_class_buttons(frame):

    classes_names = ["Low", "Medium", "High"]

    buttons = []
    for i in range(len(classes_names)):
        btn_text = str(i+1) + ": " + classes_names[i]
        buttons.append(
            tkinter.Button(frame, text=btn_text, fg="black").pack(side = "left")
        )

    return


def load_image_files(folder):
    img_ext = ['png']
    images_files = [folder + file for file in os.listdir(folder) if any(file.endswith(ext) for ext in img_ext)]
    return images_files


if __name__ == "__main__":

    window = tkinter.Tk()
    window.title("Labeling")

    # frames and buttons
    top_frame = tkinter.Frame(window).pack(side="top")
    bottom_frame = tkinter.Frame(window).pack(side="bottom")
    create_class_buttons(top_frame)

    # load images
    IMG_FOLDER = os.environ["MFP_IMG_FOLDER"]
    image_files = load_image_files(IMG_FOLDER)
    print(image_files[0])

    # show image
    img = tkinter.PhotoImage(file=image_files[0])
    label = tkinter.Label(bottom_frame, image=img).pack()


    window.mainloop()