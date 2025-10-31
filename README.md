# dms
drought monitoring system developed using Solara web apps architecture 
This application is a dynamic, web-based dashboard designed to generate a comprehensive drought bulletin for South Africa. It utilizes the Solara framework for a rich, interactive user interface and FastAPI for a robust, asynchronous backend. The system integrates data from both Google Earth Engine (GEE) for custom analysis and the ASIS Esri Image Server for official FAO products.

Functionalities and Features
The application provides end-to-end functionality for drought monitoring and reporting:

Dual-Source Geospatial Data:

 Google Earth Engine (GEE): Used for on-the-fly calculation of indices like Standardized Precipitation Index (SPI-3), and for generating NDVI, VCI, TCI, and VHI from raw MODIS and CHIRPS data based on a user-defined baseline.

 ASIS Esri Image Server: Allows users to visualize and include official, pre-calculated monthly VCI and VHI data layers from FAO's ASIS platform.



Interactive Geospatial Visualization:

 Allows users to visualize various drought indices and land cover data for South Africa on an interactive map using ipyleaflet and geemap.

 The map display dynamically switches between GEE-calculated layers and ASIS image overlays based on user selection.

Geospatial Analysis and Data Extraction:

  Enables users to define or select a Region of Interest (ROI) either through administrative boundaries (Province/District) or by uploading custom files (Shapefile, GeoJSON, KML).

  Performs robust time-series extraction and statistical summarization from GEE datasets for small ROIs (under 25 km²).

PDF Bulletin Generation:

  Generates a fully formatted, multi-page PDF drought bulletin report for the selected region.

  The report includes maps of selected indices (from GEE or ASIS), time-series charts (for small ROIs), and essential metadata (analysis period, baseline).

  Uses the fpdf library for professional PDF output, which is immediately made available for download.

Modern Web Stack:

  The frontend is built with Solara, enabling a complex, interactive UI using only Python components.

  The backend is powered by FastAPI, serving as a high-performance HTTP and WebSocket server for map interactivity and data processing requests.
