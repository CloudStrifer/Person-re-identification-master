"""Thirty neutral, conditional-slot pedestrian caption templates.

Every template uses only extracted values. The introductions establish no age,
gender, action, clothing, or setting. Missing slots disappear during rendering.
"""

from dataclasses import dataclass
import hashlib
import re
from string import Formatter


ATTRIBUTE_FIELDS = (
    "gender", "age", "upper_clothing", "lower_clothing", "footwear", "hair",
    "physical_characteristics", "clothing_type", "clothing_color",
    "clothing_material", "clothing_pattern", "accessories", "carried_items",
    "action", "posture", "background", "scene",
)
ATTRIBUTE_LABELS = {
    "gender": "gender presentation", "age": "apparent age",
    "upper_clothing": "upper clothing", "lower_clothing": "lower clothing",
    "footwear": "footwear", "hair": "hair",
    "physical_characteristics": "visible physical features",
    "clothing_type": "clothing", "clothing_color": "clothing colors",
    "clothing_material": "visible clothing material",
    "clothing_pattern": "clothing pattern", "accessories": "accessories",
    "carried_items": "carried items", "action": "action", "posture": "posture",
    "background": "background", "scene": "setting",
}
TEMPLATE_VERSION = 1


@dataclass(frozen=True)
class CaptionTemplate:
    template_id: str
    text: str

    def render(self, attributes):
        def fill(match):
            key = match.group(1)
            value = attributes.get(key)
            if not value:
                return ""
            if isinstance(value, list):
                value = ", ".join(value)
            return "{}: {}".format(ATTRIBUTE_LABELS[key], value)

        caption = re.sub(r"\{([a-z_]+)\}", fill, self.text)
        # Empty optional slots leave only separators; retain the user's values.
        prefix, details = caption.split(": ", 1)
        details = re.sub(r"(?:;\s*){2,}", "; ", details)
        details = details.strip(" ;.")
        if not details:
            raise ValueError("A caption needs at least one observed visual attribute")
        return "{}: {}.".format(prefix, details)


def _template(number, introduction, order):
    fields = tuple(order.split())
    if len(fields) != len(ATTRIBUTE_FIELDS) or set(fields) != set(ATTRIBUTE_FIELDS):
        raise ValueError("Each template must contain all optional attribute slots")
    text = introduction + ": " + "; ".join("{" + field + "}" for field in fields)
    return CaptionTemplate("visual-{:02d}".format(number), text)


# Explicit introductions and property orders, rather than random sentences that
# can introduce unobserved boots, skirts, motions, demographics, or locations.
TEMPLATES = (
    _template(1, "The image shows a person", "upper_clothing lower_clothing footwear accessories carried_items hair physical_characteristics gender age clothing_type clothing_color clothing_material clothing_pattern action posture background scene"),
    _template(2, "A person is visible in the image", "lower_clothing upper_clothing footwear carried_items accessories action posture hair physical_characteristics gender age clothing_type clothing_color clothing_material clothing_pattern scene background"),
    _template(3, "The pictured person has these visible details", "footwear upper_clothing lower_clothing accessories hair carried_items clothing_type clothing_color clothing_pattern clothing_material physical_characteristics age gender posture action background scene"),
    _template(4, "This image depicts a person", "accessories carried_items upper_clothing lower_clothing footwear clothing_color clothing_type clothing_pattern clothing_material hair age gender physical_characteristics action posture scene background"),
    _template(5, "The person in view is described by these details", "carried_items accessories footwear lower_clothing upper_clothing hair physical_characteristics gender age clothing_type clothing_color clothing_material clothing_pattern action posture scene background"),
    _template(6, "Visible features of the person include the following", "hair physical_characteristics upper_clothing lower_clothing footwear accessories carried_items age gender clothing_type clothing_color clothing_material clothing_pattern posture action background scene"),
    _template(7, "The person appears with these observed features", "physical_characteristics hair accessories carried_items upper_clothing lower_clothing footwear gender age clothing_type clothing_color clothing_pattern clothing_material action posture background scene"),
    _template(8, "A visual description of the person follows", "clothing_type upper_clothing lower_clothing clothing_color clothing_pattern clothing_material footwear accessories carried_items hair physical_characteristics gender age action posture scene background"),
    _template(9, "The image provides these details about a person", "clothing_color upper_clothing lower_clothing footwear clothing_type clothing_pattern clothing_material accessories carried_items hair physical_characteristics age gender posture action scene background"),
    _template(10, "The person's appearance is recorded as follows", "clothing_pattern clothing_material upper_clothing lower_clothing footwear accessories carried_items clothing_color clothing_type hair physical_characteristics gender age action posture background scene"),
    _template(11, "These visual attributes describe the person", "clothing_material clothing_pattern clothing_color clothing_type upper_clothing lower_clothing footwear accessories carried_items hair physical_characteristics age gender action posture scene background"),
    _template(12, "A person is pictured with the following details", "action posture upper_clothing lower_clothing footwear carried_items accessories hair physical_characteristics gender age clothing_type clothing_color clothing_material clothing_pattern background scene"),
    _template(13, "The visible person can be described as follows", "posture action carried_items accessories upper_clothing lower_clothing footwear hair physical_characteristics age gender clothing_type clothing_color clothing_material clothing_pattern scene background"),
    _template(14, "The image records these features of a person", "background scene upper_clothing lower_clothing footwear accessories carried_items hair physical_characteristics gender age clothing_type clothing_color clothing_material clothing_pattern action posture"),
    _template(15, "The pictured individual has the following attributes", "scene background carried_items accessories footwear lower_clothing upper_clothing hair physical_characteristics age gender clothing_type clothing_color clothing_pattern clothing_material posture action"),
    _template(16, "Details visible on the person are listed here", "upper_clothing clothing_color clothing_pattern clothing_material lower_clothing footwear clothing_type accessories carried_items hair physical_characteristics gender age action posture background scene"),
    _template(17, "The person is represented by these visible attributes", "lower_clothing clothing_color clothing_material clothing_pattern upper_clothing footwear clothing_type carried_items accessories hair physical_characteristics age gender action posture scene background"),
    _template(18, "The observed person has this visual description", "footwear clothing_color clothing_type upper_clothing lower_clothing accessories carried_items clothing_pattern clothing_material hair physical_characteristics gender age posture action background scene"),
    _template(19, "The following appearance details belong to the pictured person", "accessories hair carried_items physical_characteristics upper_clothing lower_clothing footwear clothing_type clothing_color clothing_material clothing_pattern gender age action posture scene background"),
    _template(20, "This is the visible description of the person", "carried_items action posture upper_clothing lower_clothing footwear accessories clothing_type clothing_color clothing_pattern clothing_material hair physical_characteristics age gender background scene"),
    _template(21, "The person shown has these recorded visual features", "hair upper_clothing clothing_type lower_clothing footwear clothing_color accessories carried_items physical_characteristics gender age clothing_material clothing_pattern action posture scene background"),
    _template(22, "The person's visible characteristics are as follows", "physical_characteristics gender age hair upper_clothing lower_clothing footwear accessories carried_items clothing_type clothing_color clothing_material clothing_pattern posture action background scene"),
    _template(23, "A person in the photograph is described through these attributes", "clothing_type clothing_material clothing_pattern clothing_color upper_clothing lower_clothing footwear hair physical_characteristics accessories carried_items age gender action posture scene background"),
    _template(24, "The following visible cues describe the person", "clothing_color accessories carried_items upper_clothing lower_clothing footwear clothing_type hair physical_characteristics gender age clothing_pattern clothing_material posture action background scene"),
    _template(25, "The person in this photograph has these observed details", "clothing_pattern upper_clothing lower_clothing clothing_type footwear clothing_material clothing_color accessories carried_items hair physical_characteristics age gender action posture scene background"),
    _template(26, "The photograph shows these attributes of a person", "clothing_material upper_clothing lower_clothing footwear clothing_pattern clothing_color clothing_type accessories carried_items hair physical_characteristics gender age posture action background scene"),
    _template(27, "The person's visual details can be stated as follows", "action carried_items upper_clothing lower_clothing footwear accessories posture hair physical_characteristics age gender clothing_type clothing_color clothing_material clothing_pattern scene background"),
    _template(28, "The individual in view has this visible description", "posture upper_clothing lower_clothing footwear action accessories carried_items hair physical_characteristics gender age clothing_type clothing_color clothing_pattern clothing_material background scene"),
    _template(29, "Here are the observed attributes of the person", "background upper_clothing lower_clothing footwear scene accessories carried_items hair physical_characteristics age gender clothing_type clothing_color clothing_material clothing_pattern posture action"),
    _template(30, "The visible individual is described by the following features", "scene upper_clothing lower_clothing footwear background carried_items accessories hair physical_characteristics gender age clothing_type clothing_color clothing_pattern clothing_material action posture"),
)

TEMPLATE_DIGEST = hashlib.sha256(
    "\n".join(template.template_id + "\t" + template.text for template in TEMPLATES).encode("utf-8")
).hexdigest()

assert len(TEMPLATES) == 30
assert len({template.template_id for template in TEMPLATES}) == 30
assert all(set(field for _, field, _, _ in Formatter().parse(template.text) if field)
           == set(ATTRIBUTE_FIELDS) for template in TEMPLATES)
