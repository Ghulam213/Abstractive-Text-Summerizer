from os import listdir
import openpyxl
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# split a document into news story and highlights
def split_story(doc):
	# find first highlight
	index = doc.find('@highlight')
	# split into story and highlights
	story, highlights = doc[:index], doc[index:].split('@highlight')
	# strip extra white space around each highlight
	highlights = [h.strip() for h in highlights if len(h)> 0]
	#print(highlights)
	return story, highlights
 
# load all stories in a directory
def load_stories(directory):
	stories = list()
	i = 0
	for name in listdir(directory):
		filename = directory + '/' + name
		if i % 10000 == 0 : print(i)
		i+=1
		# load document
		doc = load_doc(filename)
		# split into story and highlights
		story, highlights = split_story(doc)
		# store
		stories.append({'story':story, 'highlights':highlights})
		#if i == 10: break
	return stories
 
# load stories
directory = 'cnnstories'
stories = load_stories("stories")
print('Loaded Stories %d' % len(stories))

def write(stories):
        wb = openpyxl.Workbook() 
        sheet = wb.active
        sheet['A1'].value = "Stories"
        sheet['B1'].value = "Highlights"
        i = 0
        for x in range(len(stories)):
                if i % 10000 == 0 : print(i)
                i+=1
                sheet['A'+str(x+2)].value = stories[x]['story']
                highlights = ' '.join(x for x in stories[x]['highlights'])
                #print(highlights)
                sheet['B'+str(x+2)].value = highlights
        wb.save("C:\\Users\\GM\\Downloads\\cnn_stories\\cnn\\cnn.xlsx") 

write(stories)         

