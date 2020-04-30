import censusdata as cd
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd

# This script performs a principal component analysis (PCA) on 12 social
# justice factors derived from the American Community Survey (ACS). The
# study area for the analysis is Harris County, Texas. social justice
# populations were selected due to their potential effects during major
# flood events, and are assessed at the tract level. All input data are
# retrieved from the ACS API, which can be accessed through the censusdata
# Python module. The final output consists of two tables: one containing 
# the final component scores created through the PCA, and the other 
# containing the factor loadings for each component.

out_path = r'C:\Users\jj09443\working\Houston\Final Thesis Data and Docs'

# Specific fields from relevant tables are retrieved from the ACS API.

harris = cd.download(
    "acs5", 2018,
    cd.censusgeo([("state", "48"), ("county", "201"), ("tract", "*")]), 
                ["GEO_ID", 

                # Total population:
                "B01001_001E",

                # Female population:
                "B01001_026E", 

                # Under age 10:
                "B01001_003E", "B01001_004E", "B01001_027E", 
                "B01001_028E",

                # Over age 64:
                "B01001_020E", "B01001_021E", "B01001_022E",
                "B01001_023E", "B01001_024E", "B01001_025E",
                "B01001_044E", "B01001_045E", "B01001_046E",
                "B01001_047E", "B01001_048E", "B01001_049E",

                # With a disability:
                "B18101_004E", "B18101_007E", "B18101_010E",
                "B18101_013E", "B18101_016E", "B18101_019E",
                "B18101_023E", "B18101_026E", "B18101_029E",
                "B18101_032E", "B18101_035E", "B18101_038E",

                # Poverty status:
                "B17020_002E",

                # Unemploymed:
                "C18120_006E",

                # Part-time workers:
                "B23027_005E", "B23027_010E", "B23027_015E",
                "B23027_020E", "B23027_030E", "B23027_035E",

                # Renters:
                "B25009_010E",

                # Recieve public assistance:
                "B19057_002E",

                # Single parent households:
                "B09005_004E", "B09005_005E",

                # Poor English speakers:
                "C16001_005E", "C16001_008E", "C16001_011E",
                "C16001_014E", "C16001_017E", "C16001_020E",
                "C16001_023E", "C16001_026E", "C16001_029E",
                "C16001_032E", "C16001_035E", "C16001_038E",

                # No vehicle available: 
                "B08201_002E"])

# Social justice estimates are created from the retrieved fields.

harris["GEOID"] = harris["GEO_ID"].str.split("S", n=1, expand = True)[1]

harris["TOTALPOP"] = harris.B01001_001E

harris["FEMALE"] = harris.B01001_026E

harris["UNDER10"] = harris.B01001_004E + harris.B01001_003E + \
                    harris.B01001_027E + harris.B01001_028E

harris["OVER64"] = harris.B01001_020E + harris.B01001_021E + \
                    harris.B01001_022E + harris.B01001_023E + \
                    harris.B01001_024E + harris.B01001_025E + \
                    harris.B01001_044E + harris.B01001_045E + \
                    harris.B01001_046E + harris.B01001_047E + \
                    harris.B01001_048E + harris.B01001_049E

harris["DISABILITY"] = harris.B18101_004E + harris.B18101_007E + \
                        harris.B18101_010E + harris.B18101_013E + \
                        harris.B18101_016E + harris.B18101_019E + \
                        harris.B18101_023E + harris.B18101_026E + \
                        harris.B18101_029E + harris.B18101_032E + \
                        harris.B18101_035E + harris.B18101_038E

harris["POVERTY"] = harris.B17020_002E

harris["UNEMP"] = harris.C18120_006E

harris["PART_TIME"] = harris.B23027_005E + harris.B23027_010E + \
                        harris.B23027_015E + harris.B23027_020E + \
                        harris.B23027_030E + harris.B23027_035E

harris["RENTER"] = harris.B25009_010E

harris["PUB_ASSIST"] = harris.B19057_002E

harris["SINGLE_PARENT"] = harris.B09005_004E

harris["POORENG"] = harris.C16001_005E + harris.C16001_008E + \
                    harris.C16001_011E + harris.C16001_014E + \
                    harris.C16001_017E + harris.C16001_020E + \
                    harris.C16001_023E + harris.C16001_026E + \
                    harris.C16001_029E + harris.C16001_032E + \
                    harris.C16001_035E + harris.C16001_038E

harris["NOCAR"] = harris.B08201_002E

c_fields = ["FEMALE", "UNDER10", "OVER64", "DISABILITY", 
            "POVERTY", "UNEMP", "PART_TIME", "RENTER", "PUB_ASSIST",
            "SINGLE_PARENT", "POORENG", "NOCAR"]

all_fields = ["GEOID"] + c_fields

SJ_pops = pd.DataFrame(harris, columns = all_fields)

SJ_pops.reset_index(inplace=True)

# Social justice variables occur on several different scales 
# (population, households, population over 16), so they need
# to be standardized. In this script this is done through
# percentile rankings.


def percentileTable(in_data, fields):
    """This function Calculates the percentile score for each
    value in a numeric field. It takes 2 arguments:

        in_data - input pandas dataframe containing values to be 
       ranked.
    fields - list containing names of fields within dataframe
       to be ranked.

    The ouput is a pandas dataframe containing the newly created
    percentile fields.
    """
    for field in fields:
        vals = list(in_data[field])
        arr = [i for i in vals if i != 0]
        pctile = [stats.percentileofscore(
            arr, n) if n != 0 else 0 for n in vals]
        in_data["P_" + field] = pctile

    p_fields = ["P_" + field for field in fields]

    return pd.DataFrame(in_data, columns=p_fields)

# Scikit-Learn is used to perform the PCA. The PCA_kaiser function
# first scales the input values, in order to increase the
# variance within each variable. The PCA is then performed on the 
# scaled data, and components are created which are equal in number 
# to the input fields. the eigenvalues of the correlation matrix of each
# component are then assessed. The Kaiser rule is then applied to the 
# components, in which only those with an eigenvalue of 1.00 or greater 
# are retained.

def PCA_kaiser(in_data):
    """This function performs a PCA for an input dataset with multiple
    independent variables. The Kaiser rule is applied to the resulting 
    components. The output is a pandas dataframe with all remaining
    components with and eigenvalue over 1.00."""

    component_cnt = len(in_data.columns)
    X_scaled = StandardScaler().fit_transform(in_data)
    pca = PCA(component_cnt)
    f = pca.fit(X_scaled)
    t = pca.transform(X_scaled)
    PCA_Components = pd.DataFrame(t)
    keep_components = 0
    for eigval in pca.explained_variance_:
        if eigval > 1:
            keep_components = keep_components + 1
    return pd.DataFrame(PCA_Components.iloc[:, 0:keep_components])

# The output components from the PCA are rotated in order to further 
# increase the variance within the dataset.

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    """This function performs a varimax rotation for a set of PCA components.
    the input is a pandas DataFrame containing the component scores, and the
    output is a pandas DataFrame with the rotated scores"""
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(
            dot(Phi.T,asarray(
                Lambda)**3 - (gamma/p) * dot(
                    Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return pd.DataFrame(dot(Phi, R))

# Factor loadings are determined by assessing the degree of 
# correlation (positive or negative) between each component and the
# input variables. Variables with high correlation coefficients
# are considered to be the most heavily loaded. The dominant variable
# is that which has the highest absolute correlation.


def factorLoadings(PCA_table, in_table):
    """This function computes the factor loadings for each component
    in a PCA dataset. The function takes 2 arguments:

        PCA_table - dataframe containing PCA scores
        in_table - original input dataframe used to create the PCA scores
    
    The output is a dataframe containing the factor loadings for each 
    component.
    """
    compare = pd.concat([PCA_table, in_table], axis=1, sort=False)
    corr = compare.corr()
    pca_cols = len(PCA_table.columns)
    return corr.iloc[pca_cols:, :pca_cols]

percentiles = percentileTable(SJ_pops, c_fields)
in_PCA = PCA_kaiser(percentiles)
PCA_rotated = varimax(in_PCA)
fl = factorLoadings(PCA_rotated, percentiles)
# The final PCA scores and factor loadings are exported to CSV files
# in the output filepath. 

PCA_rotated.to_csv(out_path + '\PCA_rotated.csv')
fl.to_csv(out_path + '\FactorLoadings.csv')

