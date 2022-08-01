# activate acoustic_env
# streamlit run MINV_app.py

# program to perform complex matrix inverse

#pip install openpyxl
#pip install xlrd

import streamlit as st
import pandas as pd
import math
import numpy as np
import seaborn as sns
from datetime import date
import base64
sns.set_style("whitegrid")

def download_widget(object_to_download, download_file="download.csv", key=None):
    """Interactive widget to name a CSV file for download."""
    col1, col2 = st.columns(2)
    col1.write("Table shape (rows x columns):")
    col1.write(object_to_download.shape)
    filename = col2.text_input(  "Give a name to the download file", download_file, key=key)
    col2.download_button("Click to Download",object_to_download.to_csv().encode('utf-8'),filename,"text/csv",key='download-csv')

def g_fun(m,n,p,q,B,theta):
    #theta = np.arctan2(q,p)
    return np.exp(-1j*2*np.pi/B*(m*np.cos(theta)+n*np.sin(theta)))

def thetafun(q,p):
    theta = np.arctan2(q,p)
    return theta


def Read_and_Display(n,key=0):
    # Get the file
    #file1 = st.text_input("File (with path)", "None",key=key+1)
    file1 = st.file_uploader("Choose a file", "xlsx", key=1)
    
    # location of the data in the file
    sheet1 = st.text_input("Excel Sheet", "Sheet1", key=key+2)
    startingrow1 = int(st.text_input("Starting Row", 1,key=key+3))
    #matsize1 = int(st.text_input("Matrix Dimension", n**2, key=key+4))
    matsize1 = n**2
    if file1!='None':
        # read in the matrix
        #df1 = pd.read_excel(file1, sheet1, skiprows = startingrow1, usecols = list(range(0,matsize1)), index_col=None, header=None,nrows=matsize1,engine='openpyxl')
        df1 = pd.read_excel(file1, sheet1, usecols = list(range(0,matsize1)), index_col=None, header=None,nrows=matsize1,engine='openpyxl')
        # convert to numpy
        mat1 = df1.to_numpy()

        # display it ?
        #dfAB = pd.DataFrame(mat1)
        #st.write(dfAB)
        return mat1
    else:
        st.write("Matrix not yet specified.")

def main():

    B = float(st.text_input("B", 8))
    n0=1
    m0=1
    n1 = int(st.text_input("grid size", 2))
    m1 = n1
    
    thetaoption = st.selectbox(
         'Select option for theta',
         ('based on grid points', 'upload a matrix (values in radians)','upload a matrix (values in degrees)'))

    
    if thetaoption=='upload a matrix (values in radians)':
        st.header("Specify Matrix of Theta Values ("+str(n1**2)+"x"+str(n1**2)+")")
        mattheta = Read_and_Display(n1,10)
    if thetaoption=='upload a matrix (values in degrees)':
        st.header("Specify Matrix of Theta Values ("+str(n1**2)+"x"+str(n1**2)+")")
        mattheta = Read_and_Display(n1,11)
        #mattheta = mattheta.applymap(lambda z: np.radians(z))
        mattheta = np.radians(mattheta)
    theta = np.zeros((n1**2,m1**2))


    #matrix of thetas
    g = np.zeros((n1**2,m1**2),dtype=np.cdouble)
    mvals = np.zeros((n1**2,m1**2))
    nvals = np.zeros((n1**2,m1**2))
    pvals = np.zeros((n1**2,m1**2))
    qvals = np.zeros((n1**2,m1**2))

    xv = np.zeros((m1-m0+1,n1-n0+1))
    yv = np.zeros((m1-m0+1,n1-n0+1))
    for m in range(1,m1+1):
        for n in range(1,n1+1):
            for p in range(1,m1+1):
                for q in range(1,n1+1):
                    row = m-1 + (n-1)*n1
                    col = p-1 + (q-1)*n1
                    if 'upload a matrix' in thetaoption:
                        theta[row,col] = mattheta[row,col]
                    elif thetaoption=='based on grid points':
                        theta[row,col] = thetafun(q,p)
                    g[row,col]=g_fun(m,n,p,q,B,theta[row,col])
                    mvals[row,col]=m
                    nvals[row,col]=n
                    pvals[row,col]=p
                    qvals[row,col]=q

    st.header("m,n,p,q")

    mstr = np.array(["%i," % w for w in mvals.reshape(mvals.size)])
    mstr = mstr.reshape(mvals.shape)

    nstr = np.array(["%i," % w for w in nvals.reshape(nvals.size)])
    nstr = nstr.reshape(nvals.shape)

    pstr = np.array(["%i," % w for w in pvals.reshape(pvals.size)])
    pstr = pstr.reshape(pvals.shape)

    qstr = np.array(["%i" % w for w in qvals.reshape(qvals.size)])
    qstr = qstr.reshape(qvals.shape)

    tmp = np.core.defchararray.add(np.core.defchararray.add(np.core.defchararray.add(mstr, nstr), pstr), qstr)

    mnpqdf = pd.DataFrame(tmp)
    st.write(mnpqdf)


    # display the result
    st.header("theta (degrees)")
    thetadf = pd.DataFrame(theta).applymap(lambda z: np.degrees(z))
    st.write(thetadf)
    
    st.header("g")
    st.write("g(m,n,p,q)=exp(-i * 2pi / B * (m * cos(theta(p,q)) + n * sin(theta(p,q))))")
    g_df = pd.DataFrame(g).applymap(lambda z: "%0.3f %+0.3fi" % (z.real, z.imag))
    st.write(g_df)

    st.header("Condition number of g")
    gcond = np.linalg.cond(g)
    st.write(str(gcond))

    # inverse
    st.header("inv(g)")
    ginv = np.linalg.inv(g)
    g_inv = pd.DataFrame(ginv).applymap(lambda z: "%5.4g %+5.4gi" % (z.real, z.imag))
    st.write(g_inv)


    # download the results
    st.header("download g")
    download_widget(g_df,key="matrix_download1",download_file="MINV_g_%s.csv" % (str(date.today())))

    st.header("download inv(g)")
    download_widget(g_inv,key="matrix_download2",download_file="MINV_invg_%s.csv" % (str(date.today())))



if __name__ == "__main__":
    main()

