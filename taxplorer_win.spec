# -*- mode: python -*-
caffe_datas = [
    ('model/train.prototxt', './data/'),
    ('model/test.prototxt', './data/'),
    ('model/snapshot/model_iter_32264.caffemodel', './data/snapshot/'),
    ('model/TaxVocPerson_V1_1/labelmap.txt', './data/TaxVocPerson_V1_1/'),
    ('model/TaxVocPerson_V1_1/tree.txt', './data/TaxVocPerson_V1_1/'),
]

trans_datas = [
    ('../aux_data/label_to_noffset/from_coco.yaml', './label_to_noffset/'),
    ('../aux_data/label_to_noffset/from_office_v2.yaml', './label_to_noffset/'),
    ('../aux_data/label_to_noffset/from_tax4k.yaml', './label_to_noffset/'),
    ('../aux_data/label_to_noffset/from_tax700.yaml', './label_to_noffset/'),
    ('../aux_data/label_to_noffset/from_tax1300.yaml', './label_to_noffset/'),
]

block_cipher = None

a = Analysis(['gui/taxplorer.py'],
             pathex=['wincaffe'],
             binaries=[],
             datas=[ ('qt.conf', '.') ] + caffe_datas + trans_datas,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[
                '_gtkagg', '_tkagg', 'bsddb', 'curses', 'pywin.debugger', 'pywin.debugger.dbgcon', 'pywin.dialogs', 'tcl', 'Tkconstants', 'Tkinter'
             ],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='taxplorer',
          debug=False,
          strip=False,
          upx=False,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='taxplorer')
