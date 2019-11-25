from id_scripts.id_card_models import *
import id_scripts.id_card_detector
import id_scripts.id_card_reader


id_card_configurations: List[IDCardConfiguration] = []


def validate_id_card(img, validating_fields: List[ValidationField]) -> List[ValidationResult]:
    original_img = img

    detected_card_config = None
    for id_card_config in id_card_configurations:
        img = id_scripts.id_card_detector.detect_card(original_img, id_card_config)
        if img is not None:
            detected_card_config = id_card_config
            break

    if detected_card_config is None:
        raise AssertionError("Card not found!")

    results = id_scripts.id_card_reader.validate_fields(img, validating_fields, detected_card_config)
    return results
