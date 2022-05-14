"""
Taken from Pull Request #10 at nyusngroup/SESNspectraPCA repository.

This function should be a method of the SNIDsn class.
"""


def write_lnw(self, overwrite=False):
    file_lines = []
    filename = 'new_' + self.header['SN'] + '.lnw'
    header_items = []
    header_items.append('   ' + str(self.header['Nspec']))
    header_items.append(' ' + str(self.header['Nbins']))
    header_items.append('   ' + str('{:.2f}'.format(self.header['WvlStart'])))
    header_items.append('  ' + str('{:.2f}'.format(self.header['WvlEnd'])))
    header_items.append('     ' + str(self.header['SplineKnots']))
    header_items.append('     ' + str(self.header['SN']))
    header_items.append('      ' + str(self.header['dm15']))
    header_items.append('  ' + str(self.header['TypeStr']))
    header_items.append('     ' + str(self.header['TypeInt']))
    header_items.append('  ' + str(self.header['SubTypeInt']))
    header_line = ''
    for item in header_items:
        header_line += item
    file_lines.append(header_line)
    continuum = self.continuum.tolist()
    continuum_header = continuum[0]
    continuum_line = ''
    for i in range(len(continuum_header)):
        if float(continuum_header[i]) == int(continuum_header[i]):
            item = int(continuum_header[i])
        else:
            item = continuum_header[i]
        if i == 0:
            continuum_line += '     ' + str(item)
        elif i % 2 == 0:
            continuum_line += '       ' + str('{:.5f}'.format(item))
        else:
            if item >= 10:
                continuum_line += ' ' + str(item)
            else:
                continuum_line += '  ' + str(item)
    file_lines.append(continuum_line)
    continuum_all = ''
    for i in range(1, len(continuum)):
        for j in range(len(continuum[i])):
            item = str('{:.4f}'.format(continuum[i][j]))
            if j == 0:
                continuum_all += '      ' + str(i)
            else:
                if j % 2 == 0 and float(item) > 0:
                    continuum_all += '   ' + item
                else:
                    continuum_all += '  ' + item
        file_lines.append(continuum_all)
        continuum_all = ''
    phases = ['       0']
    str_phase = self.data.dtype.names
    for phase in str_phase:
        if float(phase[2:]) < 100:
            phases.append('   ' + str('{:.3f}'.format(float(phase[2:]))))
        else:
            phases.append('  ' + str('{:.3f}'.format(float(phase[2:]))))
    file_lines.append(phases)
    data = self.data.tolist()
    wvl = self.wavelengths
    count = 0
    for line in data:
        fluxes = []
        fluxes.append(' ' + str('{:.2f}'.format(wvl[count], 2)))
        for i in range(len(line)):
            if line[i] >= 0:
                fluxes.append('    ' + str('{:.3f}'.format(line[i], 3)))
            else:
                fluxes.append('   ' + str('{:.3f}'.format(line[i], 3)))
        count += 1
        file_lines.append(fluxes)
    # WFF
    filemode = "x"
    if overwrite:
        filemode = "w"

    with open(filename, filemode) as lnw:
        for line in file_lines:
            for i in range(len(line)):
                lnw.write(line[i])
            lnw.write('\n')
        lnw.close()
