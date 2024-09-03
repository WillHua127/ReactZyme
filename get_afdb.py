import torch

import argparse
from typing import List, Union
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from datetime import datetime
import time

import logging
logging.basicConfig(level=logging.INFO)



class AlphaFetcher:
	"""
	   A class to fetch and download protein metadata and files from the AlphaFold Protein Structure Database using
	   Uniprot access codes.

	   Attributes:
	       uniprot_access_list (List[str]): A list storing the Uniprot access codes to be fetched.
	       failed_ids (List[str]): A list storing any Uniprot access codes that failed to be fetched.
	       metadata_dict (dict): A dictionary storing fetched metadata against each Uniprot access code.
	       base_savedir (str): The base directory where fetched files will be saved.
   """

	def __init__(self, base_savedir=os.path.join(os.getcwd(), f'alphafetcher_results_'
	                                                          f'{datetime.now().strftime("%Y%m%d_%H%M%S")}')):
		"""
	        Initializes the AlphaFetcher class with default values.
        """
		self.uniprot_access_list = []
		self.failed_ids = []
		self.metadata_dict = {}
		self.base_savedir = base_savedir

	def add_proteins(self, proteins: Union[str, List[str]]) -> None:
		"""
	        Adds the provided Uniprot access codes to the list for fetching.

	        Args:
	            proteins (Union[str, List[str]]): A single Uniprot access code or a list of codes.

	        Raises:
	            ValueError: If the provided proteins parameter is neither a string nor a list of strings.
        """
		if isinstance(proteins, str):
			self.uniprot_access_list.append(proteins)
		elif isinstance(proteins, list):
			self.uniprot_access_list.extend(proteins)  # Using extend() method to add multiple items from a list.
		else:
			raise ValueError("Expected a string or a list of strings, but got {}".format(type(proteins)))

	def _fetch_single_metadata(self, uniprot_access: str, alphafold_database_base: str, pbar=None):
		"""
	        Fetches the metadata for a single Uniprot access code.

	        Args:
	            uniprot_access (str): The Uniprot access code to fetch.
	            alphafold_database_base (str): The base URL for the Alphafold API.
	            pbar (tqdm, optional): A tqdm progress bar. Defaults to None.
        """
		response = requests.get(f"{alphafold_database_base}{uniprot_access}")

		if response.status_code == 200:
			alphafold_data = response.json()[0]
			self.metadata_dict[uniprot_access] = alphafold_data

		else:
			self.failed_ids.append(uniprot_access)

		if pbar:
			pbar.update(1)

	def fetch_metadata(self, multithread: bool = False, workers: int = 10):
		"""
	        Fetches metadata for all the Uniprot access codes added to the class.

	        Args:
	            multithread (bool, optional): If true, uses multithreading for faster fetching. Defaults to False.
	            workers (int, optional): Number of threads to use if multithreading. If -1, uses all available CPUs.
	            Defaults to 10.
        """
		alphafold_api_base = "https://alphafold.ebi.ac.uk/api/prediction/"

		# Use all available CPUs if workers is set to -1
		if workers == -1:
			workers = os.cpu_count() or 1  # Default to 1 if os.cpu_count() returns None

		if len(self.uniprot_access_list) == 0:
			print('Please a list of Uniprot access codes with the method add_proteins()')
			return

		with tqdm(total=len(self.uniprot_access_list), desc="Fetching Metadata") as pbar:
			if multithread:
				with ThreadPoolExecutor(max_workers=workers) as executor:

					futures = [executor.submit(self._fetch_single_metadata, uniprot_access, alphafold_api_base,
					                           pbar) for uniprot_access in self.uniprot_access_list]

					# Ensure all futures have completed
					for _ in as_completed(futures):
						pass

			else:
				for uniprot_access in self.uniprot_access_list:
					self._fetch_single_metadata(uniprot_access, alphafold_api_base, pbar)

		if len(self.failed_ids) > 0:
			print(f'Uniprot accessions not found in database: {", ".join(self.failed_ids)}')

	def _download_single_protein(self, uniprot_access: str, pdb: bool = False, cif: bool = False, bcif: bool = False,
	                             pae_image: bool = False, pae_data: bool = False, pbar=None):
		"""
	       Downloads files for a single Uniprot access code.

	       Args:
	           uniprot_access (str): The Uniprot access code to fetch.
	           pdb (bool, optional): If true, downloads the pdb file. Defaults to False.
	           cif (bool, optional): If true, downloads the cif file. Defaults to False.
	           bcif (bool, optional): If true, downloads the bcif file. Defaults to False.
	           pae_image (bool, optional): If true, downloads the PAE image file. Defaults to False.
	           pae_data (bool, optional): If true, downloads the PAE data file. Defaults to False.
	           pbar (tqdm, optional): A tqdm progress bar. Defaults to None.
       """

		links_to_download = []
		metadata_dict = self.metadata_dict[uniprot_access]

		if pdb:
			pdb_savedir = os.path.join(self.base_savedir, 'pdb_files')
			extension = 'pdb'
			links_to_download.append([metadata_dict['pdbUrl'], pdb_savedir, extension])
		if cif:
			cif_savedir = os.path.join(self.base_savedir, 'cif_files')
			extension = 'cif'
			links_to_download.append([metadata_dict['cifUrl'], cif_savedir, extension])
		if bcif:
			bcif_savedir = os.path.join(self.base_savedir, 'bcif_files')
			extension = 'bcif'
			links_to_download.append([metadata_dict['bcifUrl'], bcif_savedir, extension])
		if pae_image:
			pae_image_savedir = os.path.join(self.base_savedir, 'pae_image_files')
			extension = 'png'
			links_to_download.append([metadata_dict['paeImageUrl'], pae_image_savedir, extension])
		if pae_data:
			pae_data_savedir = os.path.join(self.base_savedir, 'pae_data_files')
			extension = 'json'
			links_to_download.append([metadata_dict['paeDocUrl'], pae_data_savedir, extension])

		if len(links_to_download) == 0:
			print('Please select a type of data to download')
			return

		for data_type in links_to_download:
			data_type_url = data_type[0]
			data_type_savedir = data_type[1]
			file_extension = data_type[2]
			if not os.path.isdir(data_type_savedir):
				os.makedirs(data_type_savedir, exist_ok=True)

			response = requests.get(data_type_url)

			if response.status_code == 200:
				save_path = os.path.join(data_type_savedir, f"{uniprot_access}.{file_extension}")

				with open(save_path, 'wb') as f:
					f.write(response.content)

			else:
				print(f"Error with protein {uniprot_access}")
				return

		if pbar:
			pbar.update(1)

	def download_all_files(self, multithread: bool = False, workers: int = 10, pdb: bool = False, cif: bool = False,
	                       bcif: bool = False, pae_image: bool = False, pae_data: bool = False):
		"""
	        Downloads files for all the Uniprot access codes added to the class.

	        Args:
	            multithread (bool, optional): If true, uses multithreading for faster downloading. Defaults to False.
	            workers (int, optional): Number of threads to use if multithreading. If -1, uses all available CPUs.
	            Defaults to 10.
	            pdb (bool, optional): If true, downloads the pdb file. Defaults to False.
	            cif (bool, optional): If true, downloads the cif file. Defaults to False.
	            bcif (bool, optional): If true, downloads the bcif file. Defaults to False.
	            pae_image (bool, optional): If true, downloads the PAE image file. Defaults to False.
	            pae_data (bool, optional): If true, downloads the PAE data file. Defaults to False.
        """

		# Use all available CPUs if workers is set to -1
		if workers == -1:
			workers = os.cpu_count() or 1  # Default to 1 if os.cpu_count() returns None

		if len(self.uniprot_access_list) == 0:
			print('Please a list of Uniprot access codes with the method add_proteins()')
			return

		# This means that fetch_metadata has not been called. If it was called but had invalid codes, self.failed_ids
		# would not be empty
		if len(self.metadata_dict) == 0 and len(self.failed_ids) == 0:
			self.fetch_metadata(multithread=multithread, workers=workers)

		# This means that after fetching the metadata, there were no valid uniprot access codes
		if len(self.metadata_dict) == 0 and len(self.failed_ids) > 0:
			print('No valid Uniprot access codes provided')
			return

		valid_uniprots = self.metadata_dict.keys()
		with tqdm(total=len(valid_uniprots), desc="Fetching files") as pbar:
			if multithread:
				with ThreadPoolExecutor(max_workers=workers) as executor:
					futures = {executor.submit(self._download_single_protein, uniprot_access, pdb, cif, bcif, pae_image,
					                           pae_data, pbar): uniprot_access for uniprot_access in valid_uniprots}

					# Ensure all futures have completed and handle exceptions
					for future in as_completed(futures):
						uniprot_access = futures.get(future)
						try:
							future.result()
						except Exception as e:
							logging.error(f"Error in thread for {uniprot_access}: {e}")

			else:
				for uniprot_access in valid_uniprots:
					self._download_single_protein(uniprot_access, pdb, cif, bcif, pae_image, pae_data, pbar)
                    


def get_alphafold_download_link(uniprot_id):
    link_pattern = 'https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v2.pdb'
    return link_pattern.format(uniprot_id)

def download_alphafold_prediction(uniprot_id, path):
    url = get_alphafold_download_link(uniprot_id)
    result = subprocess.run(['wget', url, '-O', f'/{path}/{uniprot_id}.pdb'])
    return result   # Result will be 0 if operation was successful


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./data/afdb')
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print('loading...')
    trn_data = torch.load(os.path.join(args.data_dir, 'positive_train_val_time.pt'))
    tst_data = torch.load(os.path.join(args.data_dir, 'positive_test_time.pt'))
    
    uniprots = list(trn_data.keys()) + list(tst_data.keys())
    
    print(f'fetching {len(uniprots)} pdbs from alphafold database...')
    
    fetcher = AlphaFetcher(base_savedir=args.save_dir)
    # Add desired Uniprot access codes
    fetcher.add_proteins(uniprots)
    print(f'fetching {len(uniprots)} pdbs from alphafold database...')
    
    # Retrieve metadata
    fetcher.fetch_metadata(multithread=True, workers=-1)
    # Metadata available at fetcher.metadata_dict

    # Commence download of specified files
    fetcher.download_all_files(pdb=True, cif=False, multithread=True, workers=-1)
    
        
