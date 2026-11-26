# CPM published limits

`cpm23_MWARL500.csv` holds the thresholds h_n of the Mann-Whitney change point model for an in-control ARL of 500
with startup 20 (columns n, h). They are the object `cpmthresholds$MWARL500` in `R/sysdata.rda` of the R package
`cpm` 2.3 (G. J. Ross, CRAN, 8 October 2020; GPL-3), extracted unchanged. Beyond the end of the table the package
uses the mean of the last 101 values.

`scripts/cpm_hd_check.py` uses this file to check our CPM kernel. It also checks the kernel against Hawkins and Deng
(2010), J. Qual. Technol. 42(2), Table 1 (ARL0 = 500 column, p. 168); those limits are typed into the script from
the paper.
