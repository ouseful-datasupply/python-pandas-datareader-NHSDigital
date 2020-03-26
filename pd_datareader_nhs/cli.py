import pd_datareader_nhs.nhs_digital_ods as ods
import click
import sqlite3

@click.command()
@click.option('--dbname', default='nhs_ods.db',  help='SQLite database name')
@click.argument('command')
def cli(dbname,command):
	click.echo('Using SQLite3 database: {}'.format(dbname))
	if command == 'collect':
		ods.init(sqlite3db=dbname)
		ods.updatedb(sqlite3db=dbname)
