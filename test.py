import numpy as np
import random

GLO = 10


def f():
	print(GLO)


def f2():
	global GLO
	GLO = 5
	print(GLO)
	f()


f2()
# test
