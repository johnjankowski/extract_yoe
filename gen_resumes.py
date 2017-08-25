import csv
import random
import codecs
import pickle
import sys
import numpy as np
import pandas as pd


alt_titles = pd.ExcelFile('data/alternate_titles.xlsx').parse()[['Title', 'Alternate Title']]

job_title_des = pd.ExcelFile('data/occupation_data.xlsx').parse()[['Title', 'Description']]

task_des = pd.ExcelFile('data/task_statements.xlsx').parse()[['Title', 'Task']]

#tools_and_tech = pd.ExcelFile('data/tools_and_tech.xlsx').parse()[['Title', 'T2 Example']]

companies = pd.read_csv('data/us_companies.csv', usecols=['company_name'])

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

abr_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

dig_months = [str(x) for x in range(1, 13)]

abr_years = [str(x) for x in range(80, 100)] + ['0' + str(x) for x in range(10)] + [str(x) for x in range(10, 20)]

years = [str(x) for x in range(1980, 2020)]

states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
          'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 
          'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi','Missouri', 
          'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 
          'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
          'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'AL',
          'AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN',
          'MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']

csvfile = open('data/cities.csv')
reader = csv.reader(csvfile)
cities = [row[0] for row in reader]
csvfile.close()

csvfile = open('data/schools.csv')
reader = csv.reader(csvfile)
schools = [row[0] for row in reader]
csvfile.close()

csvfile = open('data/names.csv')
reader = csv.reader(csvfile)
names = [row[0] for row in reader]
csvfile.close()

last_names = np.loadtxt('data/last_names.txt', dtype='str', usecols=0).tolist()
last_names = [x[0] + x[1:].lower() for x in last_names]



degrees = ['Bachelor of Arts in English', 'Bachelor of Science in Engineering', 'MBA', 'PhD']

initials = ['A.', 'B', 'C.', 'D', 'E.', 'F', 'G.', 'H', 'I.', 'J', 'K.', 'L', 'M.', 'N', 'O.', 'P', 'Q.', 'R', 'S.', 'T.', 'U', 'V.', 'X', 'Y.', 'Z']

def gen_date():
	seed = random.random()
	if seed >= .83:
		return [random.choice(months), random.choice(years), '-', random.choice(['Current', 'Present'])]
	elif seed >= .66:
		return ['(' + random.choice(months), random.choice(years) + '-' + random.choice(months), random.choice(years) + ')']
	elif seed >= .49:
		return ['(' + random.choice(months), random.choice(years), '-', random.choice(months), random.choice(years) + ')']
	elif seed >= .33:
		return [random.choice(months), random.choice(years), '-', random.choice(months), random.choice(years)]
	elif seed >= .16:
		return [random.choice(abr_months), random.choice(abr_years), '-', random.choice(abr_months), random.choice(abr_years)]
	else:
		return [random.choice(dig_months) + '/' + random.choice(years) + 'to' + random.choice(dig_months) + '/' + random.choice(years)]

def gen_loc():
	return [random.choice(cities) + ',', random.choice(states)]

def gen_name():
	seed = random.random()
	if seed >= .5:
		return [random.choice(names), random.choice(initials), random.choice(last_names)]
	else:
		return [random.choice(names), random.choice(last_names)]


def add_work_info(resume, labels, comps, jobs, descriptions):
	seed = random.random()
	for i in range(len(comps)):
		date = gen_date()
		loc = gen_loc()
		if seed >= .83:
			resume += [comps[i].split(), date]
			resume += [jobs[i].split(), loc]
			resume += [descriptions[i].split()]
			labels += [['c' for item in comps[i].split()], ['d' for item in date]]
			labels += [['c' for item in jobs[i].split()], ['l' for item in loc]]
			labels += [['o' for item in descriptions[i].split()]]
		elif seed >= .66:
			resume += [(comps[i] + ',').split(), jobs[i].split(), date]
			resume += [descriptions[i].split()]
			labels += [['c' for item in comps[i].split()], ['c' for item in jobs[i].split()], ['d' for item in date]]
			labels += [['o' for item in descriptions[i].split()]]
		elif seed >= .49:
			resume += [comps[i].split(), loc]
			resume += [jobs[i].split(), date]
			resume += [descriptions[i].split()]
			labels += [['c' for item in comps[i].split()], ['l' for item in loc]]
			labels += [['c' for item in jobs[i].split()], ['d' for item in date]]
			labels += [['o' for item in descriptions[i].split()]]
		elif seed >= .33:
			resume += [date, jobs[i].split(), comps[i].split(), loc]
			resume += [descriptions[i].split()]
			labels += [['d' for item in date], ['c' for item in jobs[i].split()], 
					   ['c' for item in comps[i].split()], ['l' for item in loc]]
			labels += [['o' for item in descriptions[i].split()]]
		elif seed >= .16:
			resume += [comps[i].split(), jobs[i].split(), loc, date, descriptions[i].split()]
			labels += [['c' for item in comps[i].split()], ['c' for item in jobs[i].split()], ['l' for item in loc], 
					   ['d' for item in date], ['o' for item in descriptions[i].split()]]
		else:
			resume += [date, jobs[i].split()]
			resume += [comps[i].split(), loc]
			resume += [descriptions[i].split()]
			labels += [['d' for item in date], ['c' for item in jobs[i].split()]]
			labels += [['c' for item in comps[i].split()], ['l' for item in loc]]
			labels += [['o' for item in descriptions[i].split()]]
	return resume, labels


"""
FORMAT:
Phone; email
EDUCATION
University			Date
Degree, GPA			Location
(repeat above two lines)
Tools
Company 			Date
Job Title 			Location
Description
(Repeat above three lines twice)

RETURNS:
list of lists with resume content, list of lists with resume labels

Add a format where job descrtiption, date, and description are all same line (different orders)
dates in this format: (January 2012-March 2012)
"""
def gen_resume():
	my_companies = [item for sublist in companies.sample(n=3).values.tolist() for item in sublist]
	job_and_dis = job_title_des.sample(n=3)
	jobs = job_and_dis['Title'].values.tolist()
	descriptions = job_and_dis['Description'].values.tolist()
	unis = random.sample(schools, 2)
	my_degrees = random.sample(degrees, 2)
	#job_tools = [tools_and_tech.loc[tools_and_tech['Title'] == job]['T2 Example'].values.tolist() for job in jobs]
	job_tasks = [task_des.loc[task_des['Title'] == job]['Task'].values.tolist() for job in jobs]
	phone = str(random.randint(100, 999)) + '-' + str(random.randint(100, 999)) + '-' + str(random.randint(1000, 9999))
	email = str(random.randint(10000, 99999)) + '@example.com'
	resume = []
	labels = []
	resume += [[phone + ';', email], ['EDUCATION'], ['EXPERIENCE'], ['SUMMARY'], ['OBJECTIVE'], ['INTERESTS']]
	labels += [['o', 'o'], ['o'], ['o'], ['o'], ['o'], ['o'] ]
	for i in range(2):
		date = gen_date()
		resume += [unis[i].split(), date]
		labels += [['c' for item in unis[i].split()], ['d' for item in date]]
		loc = gen_loc()
		resume += [my_degrees[i].split(), ['GPA:', '3.8'], loc]
		labels += [['o' for item in my_degrees[i].split()], ['o', 'o'], ['l' for item in loc]]
	#resume += [random.sample(job_tools[0], 5), random.sample(job_tools[1], 5), random.sample(job_tools[2], 5)]
	return add_work_info(resume, labels, my_companies, jobs, descriptions)


labels = []
text = []
for i in range(1000):
	t, l = gen_resume()
	text += t
	labels += l 


with open('text.p', 'w') as f:
  pickle.dump(text, f)

with open('labels.p', 'w') as f:
  pickle.dump(labels, f)








