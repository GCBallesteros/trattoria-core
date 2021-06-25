use std::path::PathBuf;

use pyo3::prelude::*;

use tttr_toolbox::headers::File;
use tttr_toolbox::parsers::ptu::PTUFile;
use tttr_toolbox::tttr_tools::g2::{g2, G2Params};
use tttr_toolbox::tttr_tools::g3::{g3, G3Params};
use tttr_toolbox::tttr_tools::lifetime::{lifetime, LifetimeParams};
use tttr_toolbox::tttr_tools::synced_g3::{g3_sync, G3SyncParams};
use tttr_toolbox::tttr_tools::timetrace::{timetrace, TimeTraceParams};
use tttr_toolbox::tttr_tools::zero_finder::{zerofinder, ZeroFinderParams};

use ndarray::arr1;
use numpy::{IntoPyArray, PyArray1};

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::exceptions::PyTypeError;
create_exception!(trattoria_core, TrattoriaError, PyException);

// ToDo:
// - Expose better error handling instead of using a catch all

/// Parameters for the lifetime algorithm
///
/// Attributes
/// ----------
/// channel_sync: int
///     Sync channel number
/// channel_source: int
///     Source channel number
/// resolution: float
///     Resolution for the lifetime histogram in seconds
/// start_record: Optional[int]
///     First record that should be considered in the analysis
/// stop_record: Optional[int]
///     Last record that should be considered in the analysis
#[pyclass]
struct LifetimeParameters {
    #[pyo3(get)]
    channel_sync: i32,
    #[pyo3(get)]
    channel_source: i32,
    #[pyo3(get)]
    resolution: f64,
    #[pyo3(get)]
    start_record: Option<usize>,
    #[pyo3(get)]
    stop_record: Option<usize>,
}

#[pymethods]
impl LifetimeParameters {
    #[new]
    fn new(
        channel_sync: i32,
        channel_source: i32,
        resolution: f64,
        start_record: Option<usize>,
        stop_record: Option<usize>,
    ) -> Self {
        Self {
            channel_sync,
            channel_source,
            resolution,
            start_record,
            stop_record,
        }
    }
}

#[pyclass]
struct ZeroFinderParameters {
    #[pyo3(get)]
    channel_1: i32,
    #[pyo3(get)]
    channel_2: i32,
    #[pyo3(get)]
    correlation_window: f64,
    #[pyo3(get)]
    resolution: f64,
}

#[pymethods]
impl ZeroFinderParameters {
    #[new]
    fn new(channel_1: i32, channel_2: i32, correlation_window: f64, resolution: f64) -> Self {
        Self {
            channel_1,
            channel_2,
            correlation_window,
            resolution,
        }
    }
}

/// Parameters for the g2 algorithm
///
/// Attributes
/// ----------
/// channel_1: int
///     First channel
/// channel_2: int
///     Second channel
/// correlation_window: float
///     Size of the correlation window in seconds.
/// resolution: float
///     Resolution of the g2 histogram.
/// record_ranges: Optional[List[Tuple[int, int]]]
///     List of record ranges (as tuples) that should be considered when calculatin
///     g2.
///
/// Example
/// -------
/// params = G2Parameters(0, 1, 10e-9, 64e-12, [(0, 10000)])
#[pyclass]
struct G2Parameters {
    #[pyo3(get)]
    channel_1: i32,
    #[pyo3(get)]
    channel_2: i32,
    #[pyo3(get)]
    correlation_window: f64,
    #[pyo3(get)]
    resolution: f64,
    #[pyo3(get)]
    record_ranges: Option<Vec<(usize, usize)>>,
}

#[pymethods]
impl G2Parameters {
    #[new]
    fn new(
        channel_1: i32,
        channel_2: i32,
        correlation_window: f64,
        resolution: f64,
        record_ranges: Option<Vec<(usize, usize)>>,
    ) -> Self {
        G2Parameters {
            channel_1,
            channel_2,
            correlation_window,
            resolution,
            record_ranges,
        }
    }
}

/// Parameters for the g3 algorithm
///
/// Attributes
/// ----------
/// channel_1: int
///     First channel
/// channel_2: int
///     Second channel
/// channel_3: int
///     Third channel
/// correlation_window: float
///     Size of the correlation window in seconds
/// resolution: float
///     Resolution of the g3 histogram
/// start_record: Optional[int]
///     First record that should be considered in the analysis
/// stop_record: Optional[int]
///     Last record that should be considered in the analysis
#[pyclass]
struct G3Parameters {
    #[pyo3(get)]
    channel_1: i32,
    #[pyo3(get)]
    channel_2: i32,
    #[pyo3(get)]
    channel_3: i32,
    #[pyo3(get)]
    correlation_window: f64,
    #[pyo3(get)]
    resolution: f64,
    #[pyo3(get)]
    start_record: Option<usize>,
    #[pyo3(get)]
    stop_record: Option<usize>,
}

#[pymethods]
impl G3Parameters {
    #[new]
    fn new(
        channel_1: i32,
        channel_2: i32,
        channel_3: i32,
        correlation_window: f64,
        resolution: f64,
        start_record: Option<usize>,
        stop_record: Option<usize>,
    ) -> Self {
        Self {
            channel_1,
            channel_2,
            channel_3,
            correlation_window,
            resolution,
            start_record,
            stop_record,
        }
    }
}

/// Parameters for the g3 sync algorithm
///
/// Attributes
/// ----------
/// channel_sync: int
///     First channel
/// channel_1: int
///     Second channel
/// channel_2: int
///     Third channel
/// resolution: float
///     Resolution of the g3 histogram
/// start_record: Optional[int]
///     First record that should be considered in the analysis
/// stop_record: Optional[int]
///     Last record that should be considered in the analysis
#[pyclass]
struct G3SyncParameters {
    #[pyo3(get)]
    channel_sync: i32,
    #[pyo3(get)]
    channel_1: i32,
    #[pyo3(get)]
    channel_2: i32,
    #[pyo3(get)]
    resolution: f64,
    #[pyo3(get)]
    start_record: Option<usize>,
    #[pyo3(get)]
    stop_record: Option<usize>,
}

#[pymethods]
impl G3SyncParameters {
    #[new]
    fn new(
        channel_sync: i32,
        channel_1: i32,
        channel_2: i32,
        resolution: f64,
        start_record: Option<usize>,
        stop_record: Option<usize>,
    ) -> Self {
        Self {
            channel_sync,
            channel_1,
            channel_2,
            resolution,
            start_record,
            stop_record,
        }
    }
}

/// Parameters for the intensity trace algorithm.
///
/// Attributes
/// ----------
/// resolution: f64
///     Resolution of the timetrace. Smaller resolutions allow for higher temporal
///     detail but also more statistical noise.
/// channel: Optional[int]
///     Channel we want to monitor. If None all channels are integrated together.
#[pyclass]
struct TimeTraceParameters {
    #[pyo3(get)]
    resolution: f64,
    #[pyo3(get)]
    channel: Option<i32>,
}

#[pymethods]
impl TimeTraceParameters {
    #[new]
    fn new(resolution: f64, channel: Option<i32>) -> Self {
        TimeTraceParameters {
            resolution,
            channel,
        }
    }
}

/// Trattoria core is a thin python wrapper around the tttr-toolbox crate.
///
/// For a more user friendly experience consider installing the Trattoria library
/// of which this module is its heart.
#[pymodule]
fn trattoria_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TimeTraceParameters>()?;
    m.add_class::<G2Parameters>()?;
    m.add_class::<G3Parameters>()?;
    m.add_class::<G3SyncParameters>()?;
    m.add_class::<ZeroFinderParameters>()?;
    m.add_class::<LifetimeParameters>()?;
    m.add("TrattoriaError", _py.get_type::<TrattoriaError>())?;

    /// Return a dict with the contents of the header of PicoQuant PTU file.
    ///
    /// Arguments
    /// ---------
    /// filepath: str
    ///     Path to the PTU file.
    ///
    /// Returns
    /// -------
    /// Dict[str, Tuple[Any, str]]
    #[pyfn(m, "read_ptu_header")]
    fn read_ptu_header<'py>(py: Python<'py>, filepath: &str) -> PyResult<PyObject> {
        let ptu_file = PTUFile::new(PathBuf::from(filepath))
            .map_err(|_| PyTypeError::new_err("TrattoriaError"))?;
        Ok(ptu_file.header.to_object(py).into())
    }

    #[pyfn(m, "timetrace")]
    /// Returns the intensity time trace and record number time trace for a TTTR file.
    ///
    /// For details on the algorithm visit the documentation for the tttr-toolbox
    /// crate.
    fn pytimetrace<'py>(
        py: Python<'py>,
        filepath: &str,
        params: &TimeTraceParameters,
    ) -> PyResult<(&'py PyArray1<u64>, &'py PyArray1<u64>)> {
        let filename = PathBuf::from(filepath);
        let file_extension = filename
            .extension()
            .expect("File has no extension")
            .to_str()
            .expect("File has invalid extension");
        let rparams = TimeTraceParams {
            resolution: params.resolution,
            channel: params.channel,
        };

        let tttr_file = match &file_extension[..] {
            "ptu" => File::PTU(
                PTUFile::new(filename).map_err(|_| PyTypeError::new_err("TrattoriaError"))?,
            ),
            // Error::FileNotAvailable(filename_string)
            // buffered seek
            // buffer read exact
            // invalid header
            _ => return Err(PyTypeError::new_err("TrattoriaError")),
        };
        let tt_result =
            timetrace(&tttr_file, &rparams).map_err(|_| PyTypeError::new_err("TrattoriaError"))?;

        Ok((
            arr1(&tt_result.intensity[..]).into_pyarray(py),
            arr1(&tt_result.recnum_trace[..]).into_pyarray(py),
        ))
    }

    #[pyfn(m, "g2")]
    /// Returns the second order autocorrelation between two channels in the TCSPC
    ///
    /// For details on the algorithm visit the documentation for the tttr-toolbox crate.
    fn pyg2<'py>(
        py: Python<'py>,
        filepath: &str,
        params: &G2Parameters,
    ) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<u64>)> {
        let filename = PathBuf::from(filepath);
        let file_extension = filename
            .extension()
            .expect("File has no extension")
            .to_str()
            .expect("File has invalid extension");
        let rparams = G2Params {
            channel_1: params.channel_1,
            channel_2: params.channel_2,
            correlation_window: params.correlation_window,
            resolution: params.resolution,
            record_ranges: params.record_ranges.clone(),
        };

        let tttr_file = match &file_extension[..] {
            "ptu" => File::PTU(
                PTUFile::new(filename).map_err(|_| PyTypeError::new_err("TrattoriaError"))?,
            ),
            _ => return Err(PyTypeError::new_err("TrattoriaError")),
        };
        let g2_res =
            g2(&tttr_file, &rparams).map_err(|_| PyTypeError::new_err("TrattoriaError"))?;

        Ok((
            arr1(&g2_res.t[..]).into_pyarray(py),
            arr1(&g2_res.hist[..]).into_pyarray(py),
        ))
    }

    #[pyfn(m, "g3")]
    /// Returns the third order autocorrelation between three channels in the TCSPC
    ///
    /// ## Returns
    /// The matrix with the g3 correlation grows towards the right and downwards. For
    /// a more typical view (delays growing up and to the right) you will need to do a
    /// numpy.flipud on the matrix.
    ///
    /// For details on the algorithm visit the documentation for the tttr-toolbox crate.
    fn pyg3<'py>(
        py: Python<'py>,
        filepath: &str,
        params: &G3Parameters,
    ) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<u64>)> {
        let filename = PathBuf::from(filepath);
        let file_extension = filename
            .extension()
            .expect("File has no extension")
            .to_str()
            .expect("File has invalid extension");
        let rparams = G3Params {
            channel_1: params.channel_1,
            channel_2: params.channel_2,
            channel_3: params.channel_3,
            correlation_window: params.correlation_window,
            resolution: params.resolution,
            start_record: params.start_record,
            stop_record: params.stop_record,
        };

        let tttr_file = match &file_extension[..] {
            "ptu" => File::PTU(
                PTUFile::new(filename).map_err(|_| PyTypeError::new_err("TrattoriaError"))?,
            ),
            _ => return Err(PyTypeError::new_err("TrattoriaError")),
        };
        let g3_res =
            g3(&tttr_file, &rparams).map_err(|_| PyTypeError::new_err("TrattoriaError"))?;

        Ok((
            arr1(&g3_res.t[..]).into_pyarray(py),
            //g3_res.hist.view().slice(s![..;..]).to_pyarray(py).to_owned(),
            arr1(&g3_res.hist.as_slice().unwrap()).into_pyarray(py),
            //g3_res.hist.shape()[0] as usize,
            //g3_res.hist.view().into_dyn().into_pyarray(py),
            //arr2(&g3_res.hist[..;..]).into_pyarray(py),
        ))
    }

    #[pyfn(m, "g3sync")]
    /// Returns the third order autocorrelation between three channels in the TCSPC
    ///
    /// ## Returns
    /// The matrix with the g3 correlation grows towards the right and downwards. For
    /// a more typical view (delays growing up and to the right) you will need to do a
    /// numpy.flipud on the matrix.
    ///
    /// ## More information
    /// For details on the algorithm visit the documentation for the tttr-toolbox crate.
    fn pyg3sync<'py>(
        py: Python<'py>,
        filepath: &str,
        params: &G3SyncParameters,
    ) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<u64>)> {
        let filename = PathBuf::from(filepath);
        let file_extension = filename
            .extension()
            .expect("File has no extension")
            .to_str()
            .expect("File has invalid extension");
        let rparams = G3SyncParams {
            channel_sync: params.channel_sync,
            channel_1: params.channel_1,
            channel_2: params.channel_2,
            resolution: params.resolution,
            start_record: params.start_record,
            stop_record: params.stop_record,
        };

        let tttr_file = match &file_extension[..] {
            "ptu" => File::PTU(
                PTUFile::new(filename).map_err(|_| PyTypeError::new_err("TrattoriaError"))?,
            ),
            _ => return Err(PyTypeError::new_err("TrattoriaError")),
        };
        let g3_res =
            g3_sync(&tttr_file, &rparams).map_err(|_| PyTypeError::new_err("TrattoriaError"))?;

        Ok((
            arr1(&g3_res.t[..]).into_pyarray(py),
            arr1(&g3_res.hist.as_slice().unwrap()).into_pyarray(py),
        ))
    }

    #[pyfn(m, "lifetime")]
    /// Compute the lifetime histogram.
    ///
    /// For details on the algorithm visit the documentation for the tttr-toolbox crate.
    fn pylifetime<'py>(
        py: Python<'py>,
        filepath: &str,
        params: &LifetimeParameters,
    ) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<u64>)> {
        let filename = PathBuf::from(filepath);
        let file_extension = filename
            .extension()
            .expect("File has no extension")
            .to_str()
            .expect("File has invalid extension");
        let rparams = LifetimeParams {
            channel_sync: params.channel_sync,
            channel_source: params.channel_source,
            resolution: params.resolution,
            start_record: params.start_record,
            stop_record: params.stop_record,
        };

        let tttr_file = match &file_extension[..] {
            "ptu" => File::PTU(
                PTUFile::new(filename).map_err(|_| PyTypeError::new_err("TrattoriaError"))?,
            ),
            _ => return Err(PyTypeError::new_err("TrattoriaError")),
        };
        let lifetime_res =
            lifetime(&tttr_file, &rparams).map_err(|_| PyTypeError::new_err("TrattoriaError"))?;

        Ok((
            arr1(&lifetime_res.t[..]).into_pyarray(py),
            arr1(&lifetime_res.hist[..]).into_pyarray(py),
        ))
    }

    #[pyfn(m, "zerofinder")]
    fn pyzerofinder<'py>(
        py: Python<'py>,
        filepath: &str,
        params: &ZeroFinderParameters,
    ) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<u64>)> {
        let filename = PathBuf::from(filepath);
        let file_extension = filename
            .extension()
            .expect("File has no extension")
            .to_str()
            .expect("File has invalid extension");
        let rparams = ZeroFinderParams {
            channel_1: params.channel_1,
            channel_2: params.channel_2,
            correlation_window: params.correlation_window,
            resolution: params.resolution,
        };

        let tttr_file = match &file_extension[..] {
            "ptu" => File::PTU(
                PTUFile::new(filename).map_err(|_| PyTypeError::new_err("TrattoriaError"))?,
            ),
            _ => return Err(PyTypeError::new_err("TrattoriaError")),
        };
        let zf_res =
            zerofinder(&tttr_file, &rparams).map_err(|_| PyTypeError::new_err("TrattoriaError"))?;

        Ok((
            arr1(&zf_res.t[..]).into_pyarray(py),
            arr1(&zf_res.hist[..]).into_pyarray(py),
        ))
    }

    Ok(())
}
