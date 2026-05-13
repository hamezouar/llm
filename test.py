from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

class A(ABC, BaseModel):

    s : str = Field(min_length=3, max_length=10)
    n : int = Field(ge=3, le=20)

    @abstractmethod
    def say_hello(self):
        print(f"hello {self.s}")


class B(A):
    def say_hello(self):
        super().say_hello()

z = B(s='HAMZA',n=6)
z.say_hello()
