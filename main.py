
import importlib


def main():

    selection = input('please choose function you need:\n A11: detect gender with MLP.\n A12: detect gender with CNN\n'
                      'A21 detect smile with MLP\n A22: detect smile with CNN\n B1: detect face shape\n '
                      'B2: detect eye colour\n')
    if selection == 'A11':
        print('executing gender detection based on MLP')
        import A1.detect_gender
        importlib.import_module('A1.detect_gender')
    elif selection == 'A12':
        print('executing gender detection based on CNN')
        import A1.gender_CNN
        importlib.import_module('A1.gender_CNN')
    elif selection == 'A21':
        print('executing smile detection based on MLP')
        import A2.detect_emotion
        importlib.import_module('A2.detect_emotion')
    elif selection == 'A22':
        print('executing smile detection based on CNN')
        import A2.Smile_CNN
        importlib.import_module('A2.Smile_CNN')
    elif selection == 'B1':
        print('executing face shape detection')
        import B1.face_shape_cnn
        importlib.import_module('B1.face_shape_cnn')
    elif selection == 'B2':
        print('executing eye colour detection')
        import B2.eyedetect
        importlib.import_module('B2.eyedetect')

    else:
        print('wrong input')

if __name__ == "__main__":
    main()