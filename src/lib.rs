use std::path::PathBuf;

use pyo3::prelude::*;

use tttr_toolbox::parsers::ptu::PTUFile;
use tttr_toolbox::headers::File;
use tttr_toolbox::tttr_tools::timetrace::{timetrace, TimeTraceParams};
use tttr_toolbox::tttr_tools::g2::{g2, G2Params};
use tttr_toolbox::tttr_tools::zero_finder::{zerofinder, ZeroFinderParams};

use ndarray::arr1;
use numpy::{IntoPyArray, PyArray1};

// todo:
// error handling


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
    fn new(
        channel_1: i32, channel_2: i32, correlation_window: f64, resolution: f64
    ) -> Self {
        ZeroFinderParameters {
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
///     description
/// channel_2: int
///     description
/// correlation_window: float
///     description
/// resolution: float
///     description
/// start_record: Optional[int]
///     description
/// stop_record: Optional[int]
///     description
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
    start_record: Option<usize>,
    #[pyo3(get)]
    stop_record: Option<usize>,
}

#[pymethods]
impl G2Parameters {
    #[new]
    fn new(
        channel_1: i32, channel_2: i32, correlation_window: f64,
        resolution: f64, start_record: Option<usize>, stop_record: Option<usize>
    ) -> Self {
        G2Parameters {
            channel_1,
            channel_2,
            correlation_window,
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
///     desc
/// channel: Optional[int]
///     desc
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
        TimeTraceParameters { resolution, channel }
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
    m.add_class::<ZeroFinderParameters>()?;

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
        let ptu_file = PTUFile::new(PathBuf::from(filepath)).unwrap();
        Ok(ptu_file.header.to_object(py).into())
        // Error::FileNotAvailable(filename_string)
    }

    #[pyfn(m, "timetrace")]
    /// Returns the intensity time trace and record number time trace for a TTTR file.
    ///
    /// For details on the algorithm visit the documentation for the tttr-toolbox
    /// crate.
    fn pytimetrace<'py>(py: Python<'py>, filepath: &str, params: &TimeTraceParameters) -> PyResult<(&'py PyArray1<u64>, &'py PyArray1<u64>)> {
        let filename = PathBuf::from(filepath);
        let file_extension = filename.extension().expect("File has no extension").to_str().expect("File has invalid extension");
        let rparams = TimeTraceParams{ resolution: params.resolution, channel: params.channel };

        let tttr_file = match &file_extension[..] {
            "ptu" => File::PTU(PTUFile::new(filename).unwrap()),
            // Error::FileNotAvailable(filename_string)
            // buffered seek
            // buffer read exact
            // invalid header
            _ => panic!("Unrecognized file extension"),
        };
        let tt_result = timetrace(&tttr_file, &rparams).unwrap();

        Ok((
             arr1(&tt_result.intensity[..]).into_pyarray(py),
             arr1(&tt_result.recnum_trace[..]).into_pyarray(py),
        ))
    }

    #[pyfn(m, "g2")]
    /// Returns the second order autocorrelation between two channels in the TCSPC
    ///
    /// For details on the algorithm visit the documentation for the tttr-toolbox crate.
    fn pyg2<'py>(py: Python<'py>, filepath: &str, params: &G2Parameters) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<u64>)> {
        let filename = PathBuf::from(filepath);
        let file_extension = filename.extension().expect("File has no extension").to_str().expect("File has invalid extension");
        let rparams = G2Params {
            channel_1: params.channel_1,
            channel_2: params.channel_2,
            correlation_window: params.correlation_window,
            resolution: params.resolution, 
            start_record: params.start_record,
            stop_record: params.stop_record,
        };

        let tttr_file = match &file_extension[..] {
            "ptu" => File::PTU(PTUFile::new(filename).unwrap()),
            _ => panic!("Unrecognized file extension"),
        };
        let g2_res = g2(&tttr_file, &rparams).unwrap();

        Ok(
            (
                arr1(&g2_res.t[..]).into_pyarray(py),
                arr1(&g2_res.hist[..]).into_pyarray(py),
            )
        )
    }

    #[pyfn(m, "zerofinder")]
    fn pyzerofinder<'py>(py: Python<'py>, filepath: &str, params: &ZeroFinderParameters) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<u64>)> {
        let filename = PathBuf::from(filepath);
        let file_extension = filename.extension().expect("File has no extension").to_str().expect("File has invalid extension");
        let rparams = ZeroFinderParams {
            channel_1: params.channel_1,
            channel_2: params.channel_2,
            correlation_window: params.correlation_window,
            resolution: params.resolution, 
        };

        let tttr_file = match &file_extension[..] {
            "ptu" => File::PTU(PTUFile::new(filename).unwrap()),
            _ => panic!("Unrecognized file extension"),
        };
        let zf_res = zerofinder(&tttr_file, &rparams).unwrap();

        Ok(
            (
                arr1(&zf_res.t[..]).into_pyarray(py),
                arr1(&zf_res.hist[..]).into_pyarray(py),
            )
        )
    }


    Ok(())
}

