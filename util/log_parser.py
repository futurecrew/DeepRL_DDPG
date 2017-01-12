log_file = 'output/hero_20161213_170020.log'

for one_line in open(log_file, 'r'):
    if one_line.startswith('[ Test '):
        data = one_line.split('avg score: ')[1]
        print data.split(' elapsed')[0]