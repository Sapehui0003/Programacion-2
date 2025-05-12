from bs4 import BeautifulSoup
import requests
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def scrape_glassdoor_paged_spanish(base_url, num_pages):
    """
    Realiza web scraping en múltiples páginas de Glassdoor (filtrando español) extrayendo ventajas y desventajas.

    Args:
        base_url (str): La URL base de la página de evaluaciones de Glassdoor (sin el número de página).
        num_pages (int): El número total de páginas a scrapear.

    Returns:
        pandas.DataFrame: Un DataFrame con todas las ventajas y desventajas extraídas.
    """
    all_ventajas = []
    all_desventajas = []

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    for page_num in range(1, num_pages + 1):
        url = f"{base_url}_P{page_num}.htm"
        if page_num > 1:
            url += "?filter.iso3Language=spa"
        else:
            url += "?countryRedirect=true" # Mantener el parámetro de la primera página

        logging.info(f"Scrapeando la página: {url}")
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            ventajas = [item.text.strip() for item in soup.select('[data-test="review-text-PROS"]')]
            desventajas = [item.text.strip() for item in soup.select('[data-test="review-text-CONS"]')]

            all_ventajas.extend(ventajas)
            all_desventajas.extend(desventajas)

            logging.info(f"Se extrajeron {len(ventajas)} ventajas y {len(desventajas)} desventajas de la página {page_num}.")
            time.sleep(1)  # Ser amable con el servidor

        except requests.exceptions.RequestException as e:
            logging.error(f"Error al scrapear la página {page_num} ({url}): {e}")
        except Exception as e:
            logging.error(f"Ocurrió un error al procesar la página {page_num} ({url}): {e}")
            break  # Detener si hay un error inesperado

    df = pd.DataFrame({'ventaja': all_ventajas, 'desventaja': all_desventajas})
    logging.info(f"Se extrajeron un total de {len(all_ventajas)} ventajas y {len(all_desventajas)} desventajas.")
    return df

def save_dataframe(df, filepath):
    """
    Guarda un DataFrame de pandas en un archivo CSV.

    Args:
        df (pandas.DataFrame): El DataFrame a guardar.
        filepath (str): La ruta donde guardar el archivo CSV.
    """
    try:
        df.to_csv(filepath, index=False)
        logging.info(f"DataFrame guardado exitosamente en: {filepath}")
    except Exception as e:
        logging.error(f"Error al guardar el DataFrame en {filepath}: {e}")

if __name__ == "__main__":
    base_glassdoor_url = "https://www.glassdoor.com.mx/Evaluaciones/Google-Evaluaciones-E9079"
    num_pages_to_scrape = 72
    output_filepath = "data/glassdoor_all_spanish_reviews.csv"
    all_reviews_df = scrape_glassdoor_paged_spanish(base_glassdoor_url, num_pages_to_scrape)
    if all_reviews_df is not None and not all_reviews_df.empty:
        save_dataframe(all_reviews_df, output_filepath)
    elif all_reviews_df is None:
        logging.warning("No se pudieron extraer datos.")
    else:
        logging.info("No se encontraron reseñas en las páginas especificadas.")