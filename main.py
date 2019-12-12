import cv2
import id_scripts.id_card as id_card
from id_scripts.id_card_models import *
import json


def read_config_json(config_name: str) -> IDCardConfiguration:
    with open(config_name) as json_file:
        config_obj = json.load(json_file)

    template_image = cv2.imread(config_obj["imageName"])
    (height, width) = template_image.shape[0:2]
    converted_fields = []
    for field in config_obj["fields"]:
        relative_rectangle = field["positionRect"]
        field_rectangle = Rectangle(
            left=int(relative_rectangle["start"]["x"] * width),
            top=int(relative_rectangle["start"]["y"] * height),
            right=int(relative_rectangle["end"]["x"] * width),
            bottom=int(relative_rectangle["end"]["y"] * height)
        )
        if field["type"] == "name":
            field_type = IDCardFieldTypeEnum.TEXT_NAME
        elif field["type"] == "date":
            field_type = IDCardFieldTypeEnum.TEXT_DATE
        elif field["type"] == "datagram":
            field_type = IDCardFieldTypeEnum.DATAGRAM
        elif field["type"] == "barcode":
            field_type = IDCardFieldTypeEnum.BARCODE
        elif field["type"] == "city":
            field_type = IDCardFieldTypeEnum.TEXT_CITY
        else:
            field_type = IDCardFieldTypeEnum.TEXT
        converted_fields.append(IDCardFieldDescriptor(
            key=field["key"],
            rectangle=field_rectangle,
            type=field_type,
        ))

    return IDCardConfiguration(
        template=template_image,
        fields=converted_fields,
        language="hun"
    )


def convert_dict_to_validation_list(obj) -> List[ValidationField]:
    ret = []
    for key in obj:
        ret.append(ValidationField(key, obj[key]))
    return ret

if __name__ == '__main__':
    card_configs = [read_config_json("card_config.json")]

    img = cv2.imread("badge_photo.jpg")
    validate_fields = convert_dict_to_validation_list({
        "name" : "Kovács Tamás",
        "city" : "Székesfehérvár",
        "member_since" : "2018.04.12",
        "expiration" : "2020.12.01",
        "birthdate" : "1996.08.23",
        "id_code" : "123456789012"
    })

    result = id_card.validate_id_card(img, validate_fields, card_configs)
    for item in result:
        print(item)