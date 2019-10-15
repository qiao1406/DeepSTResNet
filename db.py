import mysql.connector
import re


def count(vs):
    d = {}

    pattern = re.compile(r'["\[\]\n]')

    for v in vs:
        types = re.sub(pattern, '', v).split(',')

        for t in types:
            if t in d:
                d[t] += 1
            else:
                d[t] = 1

    return d


conn = mysql.connector.connect(user='root', password='123456', database='google_places')
cursor = conn.cursor()
cursor.execute('SELECT categories FROM Places_NYC_Google')

val = cursor.fetchall()

cursor.close()
conn.close()
