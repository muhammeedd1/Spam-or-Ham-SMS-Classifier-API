from pydantic import BaseModel , field_validator , Field



class SMSRequest(BaseModel) :
    text : str = Field(
        min_text_length = 1,
        max_text_length = 1000,
        description= "SMS text to classify."
    )


@field_validator("text")
@classmethod
def Text_must_not_be_blank(cls , value : str) -> str :

    if not value.strip() :

        raise ValueError("text must not be blank or whitespace-only")
    
    return value 


class BatchSMSRequest(BaseModel) :
    texts : list[str] = Field(
        min_length= 1,
        max_length= 150,
        description="List of SMS messages to classify."
    )


@field_validator("texts")
@classmethod
def Validate_texts(cls , value : list[str]) -> list[str] :
    for text in value :
        if not text.strip() :
            raise ValueError("text must not be blank or whitespace-only")

        if len(text) > 1000 :
            raise ValueError(
                 "Each text must be at most 1000 characters."
            )
