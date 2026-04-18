# Week 3 Deliverable: Dataset Charter and Processed Dataset V1

## 1. Project Proposal

**Domain**
Educational Philanthropy, Crowdfunding, and Social Impact.

**Problem Statement**
Many public school classroom projects in underserved areas fail to secure adequate funding because donors face "information overload" and struggle to find causes that deeply resonate with their specific values. At the same time, funds are not always distributed equitably among schools with the highest poverty levels. 

**Expected Product Question**
How can we cluster the hidden educational needs based on teachers' natural language project requests, and subsequently build a recommendation engine that matches historical donors with the most urgently needed classroom projects to maximize equitable funding? Furthermore, how does the underlying donor-school graph reveal "deserts" of funding?

**Why this dataset is suitable for the course**
The dataset fulfills all technical requirements for the second half of the course:
1. **Catalog Layer:** A vast taxonomy of classroom projects, complete with categorized subjects and poverty indexes.
2. **Feature Layer:** Unstructured text (teacher essays) perfect for Natural Language Processing (TF-IDF/Embeddings) and dimensionality reduction analysis (PCA/SVD).
3. **Interaction Layer:** Explicit donor-to-project donation records and amounts, providing the vital matrix for Collaborative Filtering.
4. **Graph Layer:** A bipartite graph connecting donors and schools, allowing for centrality analysis using graph algorithms to detect community structures and isolated nodes.

## 2. Source Inventory

* **Source URL:** [DonorsChoose Dataset Archive (Kaggle)](https://www.kaggle.com/datasets/hanselhansel/donorschoose/data) 
* **Licenses or Access Conditions:** The dataset is open for educational and data science use under Kaggle's open data license structure, provided by DonorsChoose.
* **Raw File Formats:** CSV (Comma Separated Values)
* **Estimated Size:** Several gigabytes when uncompressed (Projects alone contain over 1 million rows; donations exceed 4.6 million rows).

## 3. Schema Draft

**Entity Tables & Keys:**
* `projects`: `Project_ID` (PK), `School_ID` (FK), `Teacher_ID` (FK), `Project_Subject_Category_Tree`, `Project_Title`, `Project_Essay`, `Project_Cost`
* `donations`: `Donation_ID` (PK), `Project_ID` (FK), `Donor_ID` (FK), `Donation_Amount`, `Donation_Included_Optional_Donation`
* `donors`: `Donor_ID` (PK), `Donor_City`, `Donor_State`, `Donor_Is_Teacher`
* `schools`: `School_ID` (PK), `School_Metro_Type`, `School_Percentage_Free_Lunch` (Proxy for poverty)

**Expected Joins:**
* `donations` INNER JOIN `projects` ON `donations.Project_ID = projects.Project_ID`
* `projects` INNER JOIN `schools` ON `projects.School_ID = schools.School_ID`

## 4. Processed Dataset V1 (Ingestion Script)

The following python script constitutes our reproducible Data Engineering pipeline to construct `Dataset V1`. Due to computational constraints, we filter the dataset to a recent chronological subset of fully funded projects and non-null essays.

```python
import pandas as pd
import os

def build_dataset_v1(raw_data_path, output_path):
    """
    Ingests raw DonorsChoose files, cleans the catalog and interaction layers, 
    and outputs a manageable V1 dataset for downstream modeling.
    """
    print("Loading raw CSV files...")
    
    # Load raw data (assuming files are in the specified raw directory)
    projects_df = pd.read_csv(os.path.join(raw_data_path, 'Projects.csv'))
    donations_df = pd.read_csv(os.path.join(raw_data_path, 'Donations.csv'))
    
    print(f"Initial Projects shape: {projects_df.shape}")
    
    # 1. Cleaning & Filtering Catalog (Projects)
    # Filter out projects with missing essays or critical metadata
    projects_clean = projects_df.dropna(subset=['Project Essay', 'Project Subject Category Tree', 'Project Cost'])
    
    # Chronological filter to reduce scale: only projects from 2017 onwards
    projects_clean['Project Posted Date'] = pd.to_datetime(projects_clean['Project Posted Date'])
    projects_clean = projects_clean[projects_clean['Project Posted Date'].dt.year >= 2017]
    
    # Create normalized text feature combining Title and Essay
    projects_clean['Text_Feature'] = projects_clean['Project Title'] + " " + projects_clean['Project Essay']
    
    # Select final catalog columns
    catalog_cols = ['Project ID', 'School ID', 'Teacher ID', 'Project Posted Date', 
                    'Project Subject Category Tree', 'Project Cost', 'Project Current Status', 'Text_Feature']
    catalog_v1 = projects_clean[catalog_cols]
    
    # 2. Cleaning Interaction Layer (Donations)
    # Only keep donations linked to our filtered, recent projects
    valid_project_ids = set(catalog_v1['Project ID'])
    donations_clean = donations_df[donations_df['Project ID'].isin(valid_project_ids)]
    
    # Fill missing donation amounts with median (if any) and drop invalid donors
    donations_clean['Donation Amount'] = donations_clean['Donation Amount'].fillna(donations_clean['Donation Amount'].median())
    donations_clean = donations_clean.dropna(subset=['Donor ID'])
    
    print(f"Final Catalog V1 shape: {catalog_v1.shape}")
    print(f"Final Interaction V1 shape: {donations_clean.shape}")
    
    # Output to processed directory
    os.makedirs(output_path, exist_ok=True)
    catalog_v1.to_csv(os.path.join(output_path, 'catalog_projects_v1.csv'), index=False)
    donations_clean.to_csv(os.path.join(output_path, 'interactions_donations_v1.csv'), index=False)
    print("Dataset V1 successfully built and saved to disk.")

# Note: To run this pipeline, execute the function pointing to your local data paths
# build_dataset_v1('data/raw', 'data/processed')
```

## 5. Data Dictionary Draft

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| `Project ID` | String | Unique Alphanumeric identifier for the classroom project request. |
| `School ID` | String | Unique Alphanumeric identifier for the public school. |
| `Project Subject Category Tree` | Categorical | The academic subject relevant to the project (e.g., 'Math & Science', 'Music & The Arts'). We will one-hot encode this in Week 5. |
| `Text_Feature` | Text | The concatenated string of the project's title and the teacher's essay explaining the classroom need. |
| `Project Cost` | Numeric | The total target funding amount requested (USD). |
| `Donor ID` | String | Unique Alphanumeric identifier for the donor making the contribution. |
| `Donation Amount` | Numeric | The monetary amount given by the donor to the specific project (USD). |

## 6. Scale Analysis

* **Rows & Columns:** The raw `projects` dataset contains ~1.11 million rows and 18 columns. The raw `donations` dataset contains ~4.68 million rows and 7 columns.
* **Missingness:** Missing values are predominantly found in optional metadata fields like `Project Short Description`. The primary keys (`Project ID`, `Donor ID`) have a near 0% missingness rate. Text fields (`Project Essay`) have extreme low missingness.
* **Sparsity:** The Donor-Project interaction matrix is highly sparse. Millions of donors only make 1 or 2 interactions (Long-tail distribution). 
* **Working Subset Strategy:** To manage memory constraints and the sparsity footprint, `Dataset V1` applies a strict temporal cutoff (2017-Onwards). This successfully reduces the active dataset to **308,399 projects** and **1,202,425 donations (interactions)**, which is an ideal scale for the upcoming algorithms. We will further filter out "cold-start" donors (users with fewer than 3 interactions) during the Recommendation Engine phase (Week 10) to ensure dense matrices for Collaborative Filtering.

## 7. Ethics and Access Note

* **Where the data came from:** The data is a public open-source export curated by the DonorsChoose organization, hosted on the Kaggle platform.
* **Why the team is allowed to use it:** The dataset is explicitly published to foster data science research and analytical modeling aimed at improving outcomes for educational philanthropy. 
* **What personal-data risks exist:** Although the data is public, it involves human actors (teachers writing essays, schools in specific geolocations, and donors). There is a theoretical risk of behavioral profiling.
* **How those risks were reduced:** DonorsChoose and Kaggle have preemptively hashed and completely anonymized all Unique IDs (`Teacher_ID`, `Donor_ID`). Exact real names and contact info have been scrubbed. Our team commits to using this data strictly in aggregate. We will not use cross-referencing techniques (e.g. searching essay fragments on the web) to deanonymize teachers, schools, or students. Evaluated outputs will safely mask sensitive geographic indicators.
