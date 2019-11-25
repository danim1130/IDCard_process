import cv2
import id_scripts.id_card as id_card

if __name__ == '__main__':
    img = cv2.imread("k1r.jpg")
    validate_fields = {
        "name" : "Tóth Gellért",
        "birthdate" : "1991.01.30",
        "mother_name" : "Csapó Ildikó Mária",
        "release_date" : "2002.09.20",
        "id_number" : "8453191689",
        "birthplace" : "Budapest 11"
    }
    result = id_card.validate_id_card(img, 0, validate_fields)
    print(result)
