all: grn.gen.hfst

grn.lexc.hfst: grn.lexc; hfst-lexc grn.lexc -o grn.lexc.hfst
grn.twol.hfst: grn.twol; hfst-twolc grn.twol -o grn.twol.hfst
grn.gen.hfst: grn.lexc.hfst grn.twol.hfst; hfst-compose-intersect -1 grn.lexc.hfst -2 grn.twol.hfst -o grn.gen.hfst