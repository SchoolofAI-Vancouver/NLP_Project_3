'''
ML NLP Web App
Judge Comment Toxicity
'''

from flask import Flask, jsonify, request, render_template, flash
from wtforms import Form, TextField, TextAreaField, validators, SubmitField
import config
import re

app = Flask(__name__)

# App config
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):
    comment = TextAreaField('Comment:', validators=[validators.required()])


def check_comment_toxicity(comment):
	'''
	check comment for toxicity
	input: string
	output: value between 0 - 1 indicating toxicity
	'''
	num_toxic_words = 0

	words = comment.split(' ')
	print("Words:", words)

	for word in words:

		# keep only alphabet characters!
		word = re.sub('[^a-zA-Z]+', '', word)
		# convert to lowercase
		word = word.lower()
		if word in config.toxic_words:
			num_toxic_words += 1

	print(f"Toxic Words found:{num_toxic_words}")
	print(f"Comment Length:{len(words)}")
	score = num_toxic_words / len(words)
	return score


def set_toxicity_message(toxic_score):
	'''
	sets appropriate message for toxic score
	input: toxicity score (fraction)
	output: message string
	'''
	toxicity_message = ''

	if toxic_score < 0.25:
		toxicity_message = 'Success: Your comment is not toxic.'
	elif toxic_score >= 0.25 and toxic_score < 0.5:
		toxicity_message = 'Attention: Your comment is nearly toxic.'
	elif toxic_score >= 0.5 and toxic_score < 0.75:
		toxicity_message = 'Warning: Your comment is quite toxic!'
	else:
		toxicity_message = 'Danger: Your comment is very toxic!'

	return toxicity_message


@app.route('/', methods=['GET', 'POST'])
def hello():
	form = ReusableForm(request.form)

	print(form.errors)
	if request.method == 'POST':
		print("Form:")
		print(form)
		#name = request.form['name']
		comment = request.form['comment']
		#print(f"Name:{name}")
		print(f"Comment:{comment}")

		if form.validate():
			# Save the comment here.
			#flash('Hello ' + name + ' Comment: ' + comment)
			toxic_score = check_comment_toxicity(comment)

			toxicity_message = set_toxicity_message(toxic_score)
			print(f"Toxic Score:{toxic_score}")
			print(f"Toxic message:{toxicity_message}")

			flash(f"{toxicity_message} Toxicity Score: {toxic_score:0.2f}. " +\
				f"Your comment was: {comment}")

		else:
			flash('Error: All the form fields are required. ')

	return render_template('hello.html', form=form)


if __name__ == '__main__':
    app.run()