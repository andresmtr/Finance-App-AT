"""Authenticated workbook responses with real numeric/date cells and literal text."""
from datetime import date, datetime
from decimal import Decimal
from io import BytesIO

from django.http import HttpResponse
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter


def workbook_response(sheets, filename):
    workbook=Workbook();workbook.remove(workbook.active)
    for name,headers,rows in sheets:
        sheet=workbook.create_sheet(name[:31]);sheet.append(list(headers));sheet.freeze_panes='A2'
        for cell in sheet[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='1E293B')
        for row_index,values in enumerate(rows,2):
            values=list(values)
            sheet.append(values)
            for column,value in enumerate(values,1):
                cell=sheet.cell(row_index,column)
                if isinstance(value,str):cell.data_type='s'  # References/descriptions must never become formulas.
                elif isinstance(value,Decimal):cell.number_format='#,##0.00;[Red]-#,##0.00'
                elif isinstance(value,(datetime,date)):cell.number_format='yyyy-mm-dd'
        sheet.auto_filter.ref=sheet.dimensions
        for index,header in enumerate(headers,1):sheet.column_dimensions[get_column_letter(index)].width=min(55,max(16,len(str(header))+3))
    output=BytesIO();workbook.save(output)
    response=HttpResponse(output.getvalue(),content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition']=f'attachment; filename="{filename}"';response['Cache-Control']='private, no-store'
    return response
