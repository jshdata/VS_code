def add_number(num1, num2):
    sum_num=num1+num2
    return sum_num
    
print(add_number(10, 30))
print()

fruits = ["apple", "banana", "cherry"]

#방법 1
for fruit in fruits:
    print(fruit)
print()

#방법 2
for i in range(len(fruits)):
	print(fruits[i])

print()
class MyClass:
    def __init__(self, name="Unknown", age=0):
        self.name = name
        self.age = age

    def greet(self):
        print(f"Hello, my name is {self.name} and I am {self.age} years old.")


# 클래스 인스턴스화
person1 = MyClass()
person2 = MyClass("Alice", 25)

person1.greet()
person2.greet()