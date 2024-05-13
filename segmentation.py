import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Описание
print('\n----------------------------------')
print('Программа производит сегментацию')
print('изображения выбранным методом.')
print('----------------------------------')

# Выбор метода сегментации
while True:
    print()
    print('(1) Сегментация нижним порогом;')
    print('(2) Сегментация с использованием гистограммы;')
    print('(3) Сегментация методом водораздела.')
    method=input('Введите номер метода: ')
    if method!='1' and method!='2' and method!='3':
        print('Невернный ввод. Попробуйте снова.')
    else:
        break

# Чтение изображения
while True:
    print()
    read_path=input('Введите путь изображения: ')
    image=cv.imread(read_path)
    if image is None:
        print('Не удалось считать изображение. Попробуйте снова.')
    else:
        break
image=cv.cvtColor(image, cv.COLOR_RGB2BGR)

# Размытие
image_blur=cv.GaussianBlur(image, (5,5), 0)

# Сегментация нижним порогом
image_segmented=None
if method=='1':
    while True:
        try:
            threshold=int(input('\nВведите пороговое значение (0-255): '))
        except:
            print('Неверный ввод. Попробуйте снова.')
        else:
            if threshold<0 or threshold>255:
                print('Значение должно быть от 0 до 255. Попробуйте снова.')
            else:
                break
    image_gray=cv.cvtColor(image_blur, cv.COLOR_BGR2GRAY)
    image_segmented=cv.threshold(image_gray, threshold, 255, cv.THRESH_BINARY)[1]

# Сегментация с использованием гистограммы
elif method=='2':

    # Создание гистограммы
    print('\nВывод гистограммы изображения...')
    image_gray=cv.cvtColor(image_blur, cv.COLOR_BGR2GRAY)
    gray_hist=cv.calcHist([image_gray], [0], None, [256], [0,256])

    # Вывод гистограммы
    plt.figure('Гистограмма серых тонов')
    plt.plot(gray_hist)
    plt.title('Найдите локальные минимумы.\nВыберите их в качестве пороговых значений.')
    plt.xlabel('Яркость')
    plt.ylabel('Пиксели')
    plt.xlim([0,255])
    plt.show()

    # Ввод пороговых значений
    thresholds=list()
    while True:
        try:
            n=int(input('Введите количество пороговых значений: '))
        except:
            print('Неверный ввод. Попробуйте снова.\n')
        else:
            if n<0:
                print('Значение должно быть положительным. Попробуйте снова.\n')
            else:
                break

    print('Введите {0} пороговых значений (0-255):'.format(n))
    for i in range(n):
        while True:
            try:
                t=int(input('Порог №{0}: '.format(i+1)))
            except:
                print('Неверный ввод. Попробуйте снова.\n')
            else:
                if t<0 or t>255:
                    print('Значение должно быть от 0 до 255. Попробуйте снова.\n')
                else:
                    thresholds.append(t)
                    break

    # Создание сегментов и совмщение в целое изображение
    image_segmented=np.zeros(image.shape[:2], dtype='uint8')
    for i in range(n):
        gray_level=int((i+1)/n*255)
        segment=cv.threshold(image_gray, thresholds[i], 255, cv.THRESH_BINARY)[1]
        image_segmented[segment==255]=gray_level

#Сегментация методом водораздела    
else:

    # Пороговая сегментация
    image_gray=cv.cvtColor(image_blur, cv.COLOR_BGR2GRAY)
    image_bin=cv.threshold(image_gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    
    # Преобразование в расстояние
    image_D=cv.distanceTransform(image_bin, cv.DIST_L2, 5)
    
    # Выделение маркеров
    image_max=cv.threshold(image_D, 0.7*image_D.max(), 255, cv.THRESH_BINARY)[1]
    image_max=np.uint8(image_max)
    unknown=cv.bitwise_xor(image_bin, image_max)
    markers=cv.connectedComponents(image_max)[1]
    markers+=1
    markers[unknown==255]=0
    
    # Применение метода водораздела
    image_segmented=markers.copy()
    cv.watershed(image_blur, image_segmented)

# Вывод сегментированного изображения
print('\nВывод сегментированного изображения...')
fig, axs=plt.subplots(1,2)
fig.canvas.manager.set_window_title('Результат сегментации')
axs[0].set_title('Оригинал')
axs[0].imshow(image)
axs[1].set_title('Сегментация')
axs[1].imshow(image_segmented, cmap='jet')
plt.show()

# Запись сегментированного изображения
while True:
    is_saving=input('\nСохранить сегментированное изображение? (Y)Да/(N)Нет: ')
    is_saving=is_saving.lower()
    if is_saving=='y':
        while True:
            write_path=input('Введите путь сохранения: ')
            try:
                cv.imwrite(write_path, np.uint8(image_segmented))
            except:
                print('Неверный ввод. Попробуйте снова.\n')
            else:
                print('Изображение сохранено.')
                break
        break
    elif is_saving=='n':
        break
    else:
        print('Невернный ввод. Попробуйте снова.')
