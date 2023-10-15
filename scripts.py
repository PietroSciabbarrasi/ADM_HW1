#Say "Hello, World!" With Python
if __name__ == '__main__':
    my_string="Hello, World!"
    print(my_string)

#Python If-Else
import math
import os
import random
import re
import sys
if __name__ == '__main__':
    n = int(input().strip())
if n%2 == 1:
    print("Weird")
if (n%2 == 0) and (2<=n<=5):
    print("Not Weird")
if (n%2 == 0) and (6<=n<=20):
    print("Weird")
if (n%2==0) and (n>20):
    print("Not Weird")

#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a//b)
print(a/b)


#Loops
if __name__ == '__main__':
    n = int(input())
for i in range(n):
    print(i*i)

#Write a function
def is_leap(year):
    leap = False
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        leap=True
    
    return leap

#Print Function
if __name__ == '__main__':
    n = int(input())
string=""
for i in range(1,n+1):
    string += str(i)
print(string)

#Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
arr=list(arr)
arr.sort(reverse=True)
classifica=list(set(arr))
print(classifica[1])

#List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    lista = [[i, j, k] for i in [ i for i in range(x+1) ] for j in [ j for j in range(y+1) ] for k in [ k for k in range(z+1) ]]
    out=[]
    for i in lista:
        if sum(i)!=n:
            out.append(i)
    print(out)

#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
num=(sum(student_marks[query_name]))/len(student_marks[query_name])
num = "{:.2f}".format(num)
print(num)

#Nested Lists
if __name__ == '__main__':
    lista=[]
    punti=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        punti.append(score)
        app=[name,score]
        lista.append(app)
    punti.sort()
    single_scores=list(set(punti))
    second=single_scores[1]
    out=[]
    for i in lista:
        if i[1]==second:
            out.append(i[0])
    for i in sorted(out):
        print(i)

#Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
t=tuple(integer_list)
print(hash(t))

#Lists
if __name__ == '__main__':
    N = int(input())
lista=[]    
for i in range(N):
    lista.append(input())
listone=[]
lista1=[]
for i in lista:
    i=i.split()
    if i[0]=="insert":
        lista1.insert(int(i[1]),int(i[2]))
    if i[0]=="remove":
        lista1.remove(int(i[1]))
    if i[0]=="append":
        lista1.append(int(i[1]))
    if i[0]=="sort":
        lista1.sort()
    if i[0]=="reverse":
        lista1.reverse()
    if i[0]=="pop":
        lista1.pop()
    if i[0]=="print":
        print(lista1)

#sWAP cASE
def swap_case(s):
    stringa=""
    for i in s:
        if i.isupper()==True:
            stringa+=i.lower()
        else:
            stringa+=i.upper()
    
    return(stringa)

#String Split and Join
def split_and_join(line):
    lis=line.split()
    lis="-".join(lis)
    return(lis)

#What's Your Name?
def print_full_name(first, last):
    sol="Hello "+first+" "+last+"! You just delved into python."
    print(sol)

#Mutations
def mutate_string(string, position, character):
    sol=list(string)
    sol[position]=character
    sol="".join(sol)
    return(sol)

#Find a string
def count_substring(string, sub_string):
    counter=0
    for i in range(0,len(string)-len(sub_string)+1):
        if string[i:i+len(sub_string)]==sub_string:
            counter = counter+1
    return counter

#String Validators
if __name__ == '__main__':
    s = input()
alfanum=False
alfa=False
dig=False
lower=False
upper=False
for i in s:
    if i.isalnum():
        alfanum=True
    if i.isalpha():
        alfa=True
    if i.isdigit():
        dig=True
    if i.islower():
        lower=True
    if i.isupper():
        upper=True
print(alfanum)
print(alfa)
print(dig)
print(lower)
print(upper)

#Text Alignment
thickness = int(input()) 
c = 'H'
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#Text Wrap
def wrap(string, max_width):
    sol=textwrap.wrap(string,max_width)
    out=""
    for i in sol:
        out+=i+"\n"
    return(out)

#String Formatting
def print_formatted(number):
    D=[]
    O=[]
    H=[]
    B=[]
    for i in range(1,number+1):
        D.append(i)
    for i in range(1,number+1):
        O.append(oct(i)[2:])
    for i in range(1,number+1):
        H.append(hex(i)[2:].upper())
    for i in range(1,number+1):
        B.append(bin(i)[2:])
    sol=""
    space=(len(B[len(B)-1]))
    for i in range(len(D)):
        sol+=" "*(space-len(str(D[i])))+str(D[i])+" "+" "*(space-len(str(O[i])))+O[i]+" "+" "*(space-len(H[i]))+H[i]+" "+" "*(space-len(B[i]))+B[i]+"\n"
        
    print(sol)

#Introduction to Sets
def average(array):
    return sum(set(array))/len(set(array))

#Symmetric Difference
M = int(input())
a = set(input().split())
N = int(input())
b = set(input().split())
sol=[]
set_sol=(a.union(b))-(a.intersection(b))
for i in set_sol:
    sol.append(int(i))

for i in sorted(sol):
    print(i)

#Set .add()
N = int(input())
stati = set()
for i in range(N):
    stati.add(input())
print(len(stati))

#Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
n_command=int(input())
commands=[]
for i in range(n_command):
    commands.append(input())
    
for i in commands:
    o=i.split()
    if o[0]=="pop":
        s.pop()
    if o[0]=="remove":
        s.remove(int(o[1]))
    if o[0]=="discard":
        s.discard(int(o[1]))
    
print(sum(s))

#Set .union() Operation
n_eng=int(input())
eng=set(input().split())
n_fra=int(input())
fra=set(input().split())

print(len(fra.union(eng)))

#Set .intersection() Operation
n_eng=int(input())
eng=set(input().split())
n_fra=int(input())
fra=set(input().split())
print(len(fra.intersection(eng)))

#Set .difference() Operation
n_eng=int(input())
eng=set(input().split())
n_fra=int(input())
fra=set(input().split())
print(len(eng.difference(fra)))

#Set .symmetric_difference() Operation
n_eng=int(input())
eng=set(input().split())
n_fra=int(input())
fra=set(input().split())
print(len(fra.symmetric_difference(eng)))

#Set Mutations
n_a=int(input())
a=set(input().split())
n_set=int(input())
for i in range(1,n_set+1):
    txt= input().split()
    insieme = set(input().split())
    
    if txt[0] == 'update':
        a.update(insieme)
    elif txt[0] == 'intersection_update':
        a.intersection_update(insieme)
    elif txt[0] == 'difference_update':
        a.difference_update(insieme)
    else:
        a.symmetric_difference_update(insieme)
sol=[]
for i in a:
    sol.append(int(i))
print(sum(sol))

#Check Subset
X= int(input())
for i in range(X):
    nA = int(input())
    A = set(map(int, input().split()))
    nB = int(input())
    B = set(map(int, input().split()))
    Sol = A.difference(B)
    if len(Sol) == 0:
        print("True")
    else:
        print("False")

#Designer Door Mat
n,m = map(int, input().split())
meta = int(n/2)
for i in range(n):
    if i < meta:
        print(('.' + ('|..'*i) + '|' + ('..|'*i) + '.').center(m,'-'))
    elif i == meta:
        print('WELCOME'.center(m,'-'))
    else:
        print(('.' + ('|..'*(n-i-1)) + '|' + ('..|'*(n-i-1)) + '.').center(m,'-'))

#Alphabet Rangoli
def print_rangoli(size):
    alfabeto = "abcdefghijklmnopqrstuvwxyz"
    w = (size*4)-3
    rows = (size*2)-1
    for i in range(0, rows):
        index = abs(size-(i+1))
        n = size-(index+1)
        m = (n*2)+1
        a = ''
        for j in range(m):
            c = abs(n-j)+index
            a = '-'.join([alfabeto[c],a])
        print(a.center(w, '-')[:w])

#Check Strict Superset
A = set(map(int, input().split()))
n = int(input())
sol = []
for i in range(n):
    A2 = set(map(int, input().split()))
    dif = A2.difference(A)
    sol += dif
if len(sol) == 0:
    print("True")
else:
    print("False")

#Arrays
import numpy
def arrays(arr):
    arr = numpy.array(arr,float)
    return numpy.flip(arr)

#Shape and Reshape
import numpy
x=list((map(int, input().split())))
arr=numpy.array(x)
arr.shape=(3,3)
print(arr)

#Transpose and Flatten
import numpy
N, M = map(int,input().split())
arr = []
for i in range(N):
    el = list(map(int, input().split()))
    arr.append(el)   
arr = numpy.array(arr)
print(numpy.transpose(arr))
print(arr.flatten())

#Concatenate
import numpy
N,M,P=map(int,input().split())
p=[]
m=[]
for i in range(N):
    p.append(list(map(int, input().split())))
for i in range(M):
    m.append(list(map(int, input().split())))
arr1 = numpy.array(p)
arr2 = numpy.array(m)
print(numpy.concatenate((arr1, arr2), axis=0))

#Zeros and Ones
import numpy
N = list(map(int,input().split()))
zeri = numpy.zeros((N),int)
uni = numpy.ones((N),int)
print(zeri,uni,sep = "\n")

#Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
A = numpy.array(input().split(), float)
print(numpy.floor(A), numpy.ceil(A), numpy.rint(A), sep='\n')

#Min and Max
import numpy
N,M=map(int,input().split())
arr=[]
for i in range(N):
    arr.append(list(map(int,input().split())))  
arr=numpy.array(arr)
print(numpy.max(numpy.min(arr,axis=1)))

#Dot and Cross
import numpy
n=int(input())
A = []
B=[]
for i in range(n):
    A.append(list(map(int, input().split())))
for i in range(n):
    B.append(list(map(int, input().split())))
A = numpy.array(A,int)
B = numpy.array(B,int)
print(numpy.matmul(A, B))

#Polynomials
import numpy
P = list(map(float,input().split()))
x = int(input())
print(numpy.polyval(P,x))

#Inner and Outer
import numpy
A = numpy.array(input().split(),int)
B = numpy.array(input().split(),int)
print(numpy.inner(A,B))
print(numpy.outer(A,B))

#Sum and Prod
import numpy
n, i = map(int, input().split())
A = []
for i in range(n):
    A.append(list(map(int, input().split())))
print(numpy.prod(numpy.sum(A, axis=0), axis=0))

#Eye and Identity
import numpy
numpy.set_printoptions(legacy='1.13')
N,M=map(int,input().split())
print(numpy.eye(N,M))

#Array Mathematics
import numpy
N,M = (int(i) for i in input().split())
arr_A = numpy.array([[int(i) for i in input().split()] for j in range(N)])
arr_B = numpy.array([[int(i) for i in input().split()] for j in range(N)])
print(numpy.add(arr_A,arr_B))
print(numpy.subtract(arr_A,arr_B))
print(numpy.multiply(arr_A,arr_B))
print(arr_A//arr_B)
print(numpy.mod(arr_A,arr_B))
print(numpy.power(arr_A,arr_B))

#Linear Algebra
import numpy
N = int(input())
A = numpy.array([input().split() for j in range(N)],float)
print(round(numpy.linalg.det(A),2))

#Mean, Var, and Std
import numpy
N, M = map(int, input().split())
A = numpy.array([input().split(" ") for j in range(N)], int)
print(numpy.mean(A, axis=1))
print(numpy.var(A, axis=0))
print(round(numpy.std(A, axis=None), 11))

#Birthday Cake Candles
import math
import os
import random
import re
import sys
def birthdayCakeCandles(candles):
    m=max(candles)
    count=0
    for i in candles:
        if i==m:
            count+=1
    return(int(count))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

#Number Line Jumps
import math
import os
import random
import re
import sys
def kangaroo(x1, v1, x2, v2):
    if v1 <=v2:
        return "NO"
    elif v1>v2:
        if(x2-x1)%(v1-v2)==0: 
            return "YES"
        else: return "NO"
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

#Viral Advertising
import math
import os
import random
import re
import sys
def viralAdvertising(n):
    sum=0
    x=5
    for i in range(n):
        sum+=x//2
        x=(x//2)*3
    return sum
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

#Re.start() & Re.end()
import re
S, k = input(), input()
x = 0
m = re.search(k, S[x:])
if not m:
    print("(-1, -1)") 
else:
    while m:
        print(f"({x + m.start()}, {x + m.end() - 1})")
        x = x + m.start() + 1
        m = re.search(k, S[x:])

#Regex Substitution
import re
def cambia(testo):
    return {'&&': 'and', '||': 'or'}[testo.group(0)]
for i in range(int(input())):
    print(re.sub(r'(?<=[ ])([&|])\1(?=[ ])', cambia, input()))

#Validating Roman Numerals
regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

#Validating phone numbers
import re
n=int(input())
for i in range(n):
    print('YES' if re.match(r'^[7-9]\d{9}$', input()) else 'NO')

#Validating and Parsing Email Addresses
import re
import email.utils
n=int(input())
pattern = r"^([a-zA-Z][a-zA-Z0-9\_\-\.]+)\@[a-zA-Z]+\.[a-zA-Z]{1,3}$"
for i in range(n):
    name, email_address = email.utils.parseaddr(input())
    if re.search(pattern, email_address):
        print(email.utils.formataddr((name, email_address)))

#Detect Floating Point Number
import re
T = int(input())
for i in range(T):
    x=re.match(r"^[+-]?\d*\.\d+$",input())
    print(bool(x))

#Re.split()
regex_pattern = r"[,.]"

#Group(), Groups() & Groupdict()
import re
S = input()
m = re.search(r"([a-zA-Z0-9])\1", S)
if m:
    print(m.group(1))
else:
    print(-1)

#Re.findall() & Re.finditer()
import re
S = input()
x=re.findall(r"\B[AEIOUaeiou]{2,}\B[^AEIOUaeiou]",S) 
if x :
    for i in x:
        print(i[:-1])
else :
    print(-1)

#Hex Color Code
import re
n=int(input())
for i in range(n):
    S = input()
    x=re.search(r'^\s.*(#[\da-fA-F]{3,6})', S)
    if x:
        print(*re.findall(r'#[\da-fA-F]{3,6}', S), sep='\n')

#HTML Parser - Part 1
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for name,value in attrs:
            print(f"-> {name} > {value}")
    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for name,value in attrs:
            print(f"-> {name} > {value}")
    def handle_endtag(self, tag):
        print(f"End   : {tag}")
parser = MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())
parser.close()

#HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if '\n' in data:
            print('>>> Multi-line Comment')
            print(data)
        else:
            print('>>> Single-line Comment')
            print(data)
    def handle_data(self,data):
        if '\n' not in data:
            print('>>> Data')
            print(data)
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
parser = MyHTMLParser()
parser.feed(html)
parser.close()

#Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for i in attrs:
            print('->', i[0], '>', i[1])
parser = MyHTMLParser()
N = int(input())
for i in range(N):
    parser.feed(input())

#Validating UID
import re
x = r'^(?=(?:.*[A-Z]){2,})(?=(?:.*\d){3,})(?!.*(.).*\1)[\w]{10}$'
for i in range(int(input())):
    UID = input()
    if re.match(x, UID):
        print("Valid")
    else:
        print("Invalid")

#The Minion Game
def minion_game(string):
    stuart = 0
    kevin = 0
    for x in range(len(string)):
        if string[x] in "AEIOU":
            kevin += len(string)-x
        else:
            stuart += len(string)-x 
    if stuart > kevin:
        print(f"Stuart {stuart}")
    elif stuart < kevin:
        print(f"Kevin {kevin}")
    else:
        print("Draw")

#Merge the Tools
def merge_the_tools(string, k):
    s = int(len(string)/k)
    for i in range(s):
        t = ""
        new = set()
        for j in string[i * k: (i + 1) * k]:
            if j not in new:
                t += j
                new.add(j)   
        print(t)

#The Captain's Room
k = int(input())
rooms = input().split()
A = set(rooms)
for i in A:
    rooms.remove(i)
print((A ^ set(rooms)).pop())

#Validating Credit Card Numbers
import re 
pattern1 = r'[456]\d{3}\-?\d{4}\-?\d{4}\-?\d{4}'
pattern2 = r'(\d)\1{3,}'
n=int(input())
for i in range(n):
    Carta = input()
    match = re.fullmatch(pattern1, Carta) 
    if match:
        x = re.search(pattern2, Carta.replace('-','')) 
        if x:
            print('Invalid')
        else:
            print('Valid') 
    else:
        print('Invalid')

#collections.Counter()
import collections
X=int(input())
scarpe = collections.Counter(map(int, input().split()))
sol = 0
N=int(input())
for i in range(N):
    taglia, prezzo = map(int, input().split())
    if scarpe[taglia]:
        sol += prezzo
        scarpe[taglia] -= 1
print(sol)

#DefaultDict Tutorial
from collections import defaultdict 
n,m = map(int,input().split())
d = defaultdict(list)
for i in range(n):
    d[input()].append(str(i+1))
for i in range(m):
    B = input()
    if B in d:
        print(" ".join(map(str, d[B])))
    else:
        print(-1)

#Collections.namedtuple()
from collections import namedtuple
num = int(input())
Studenti = namedtuple("studenti", input().split())
print(sum([int(Studenti(*(input().split())).MARKS) for n in range(num)])/num)

#Collections.OrderedDict()
from collections import OrderedDict
N = int(input())
dic = OrderedDict()
for _ in range(N):
    item_name, item_price = input().rsplit(' ', 1)
    if item_name in dic:
        dic[item_name] += int(item_price)
    else:
        dic[item_name] = int(item_price)  
for item_name, net_price in dic.items():
    print(item_name, net_price)

#Collections.deque()
from collections import deque 
n=int(input()) 
d=deque()
a=[] 
for i in range(n): 
    a.append(list(map(str,input().split())))
for j in a:
    if j[0]=='append':
        d.append(int(j[1]))
    elif j[0]=='pop':
        d.pop()
    elif j[0]=='popleft':
        d.popleft()
    elif j[0]=='appendleft':
        d.appendleft(int(j[1]))        
print(*d)

#Word Order
from collections import Counter
n = int(input())
l = []
for i in range(n):
    l.append(input())
count = list(Counter(l).values())
print(len(count))
print(*count)

#Piling Up!
T=int(input())
for i in range(T):
    n=int(input())
    l=list(map(int,input().split()))
    count=0
    for i in range(n//2):
        b1=max(l[i],l[n-1-i])
        b2=max(l[i+1],l[n-2-i])
        if(b1<b2):
            print('No')
            count=1
            break                
    if(count==0):
        print('Yes')

#Company Logo
import math
import os
import random
import re
import sys
from collections import Counter 
if __name__ == '__main__':
    s = input()
    s1 = sorted(s)
    d = Counter(s1)
    d1 = d.most_common(3)
    for i,j in d1:
        print('{} {}'.format(i,j))

#Validating Postal Codes
regex_integer_in_range = r"^[1-9]([0-9]){5}$"
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"

#Matrix Script
import math
import os
import random
import re
import sys
first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])
matrix = []
s = ''
for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
for i in range(m):
    for j in range(n):
        s += matrix[j][i]
print(re.sub(r'\b(\W)+\b', ' ', s))

#Calendar Module
import calendar
MM, DD, YY = map(int, input().split())
day = calendar.weekday(YY, MM, DD)
days = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
print(str((days[day])).upper())

#Time Delta
import math
import os
import random
import re
import sys
from dateutil import parser
def time_delta(t1, t2):
    delta = parser.parse(t1) - parser.parse(t2)
    return "{:.0f}".format(abs(delta.total_seconds()))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')
    fptr.close()

#Exceptions
T = int(input())
for _ in range(T):
    try:
        values = input()
        a = int(values.split(" ")[0])
        b = int(values.split(" ")[1])
        print(a//b)
    except (ZeroDivisionError, ValueError) as e:
        print(f"Error Code: {e}")

#Zipped!
N, X = map(int, input().split())
lista = []
for i in range(X):
    lista.append(list(map(float, input().split())))
studenti = list(zip(*lista))
for j in studenti:
    print(sum(j)/X)

#Athlete Sort
import math
import os
import random
import re
import sys
if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())
    arr.sort(key=lambda row: row[k])
    for row in arr:
        print(*row)

#Map and Lambda Function
cube = lambda x: x**3
def fibonacci(n):
    if n==0:
        return []
    if n==1:
        return [0]
    if n==2:
        return [0,1]
    x=[0, 1]
    while len(x)<n:
        x.append(x[-1]+x[-2])
    return x

#XML 1 - Find the Score
def get_attr_number(node):
    count=0
    for i in node.iter():
        count+=len(i.attrib)
    return count

#XML2 - Find the Maximum Depth
maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    if level >= maxdepth:
        maxdepth = level
    for i in elem:
        depth(i, level)

#Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        sol = []
        for i in l:
            sol.append('+91 '+i[-10:-5]+' '+ i[-5:])
        f(sol)
    return fun

#Decorators 2 - Name Directory
def person_lister(f):
    def inner(people):
        return ([f(p) for p in sorted(people, key=lambda p: int(p[2])) ])
    return inner

#Recursive Digit Sum
import math
import os
import random
import re
import sys
def superDigit(n, k):
    sol=0
    if len(n)==1:
        return int(n)
    else:
        for i in n:
            sol=sol+int(i)
        sol=sol*k
        return superDigit(str(sol),1)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()

#Insertion Sort - Part 1
import math
import os
import random
import re
import sys
def insertionSort1(n, arr):
    length = len(arr)
    t = arr[length - 1]
    for i in range(length, 0, -1):
        if t < arr[i - 2]:
            if arr[i - 2] == arr[-1]:
                arr[i - 1] = t
                for j in arr:
                    print(j, end = " ")
                print(end = "\n")
            else:
                arr[i - 1] = arr[i - 2]
                for j in arr:
                    print(j, end = " ")
                print(end = "\n")
        else:
            arr[i - 1] = t
            for j in arr:
                print(j, end = " ")
            break
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

#Insertion Sort - Part 2
import math
import os
import random
import re
import sys
def insertionSort2(n, arr):
    for i in range(1, n):
        x = arr[i]
        count = 0
        if x < arr[i-1]:
            for j in range(i, 0, -1):
                if x < arr[j-1]:
                    arr[j] = arr[j-1]
                    arr[j-1] = x
                else:
                    break
        print(" ".join(map(str, arr)))
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)