import React from 'react';
import ReactDOM from 'react-dom';
import routes  from './routes/router';
import { BrowserRouter } from 'react-router-dom';
import { renderRoutes } from 'react-router-config';

import './index.css';
// import App from './App';
import * as serviceWorker from './serviceWorker';
// document.body.addEventListener('touchmove', function (e) {
//     e.preventDefault(); 
// }, {passive: false}); 
ReactDOM.render( <BrowserRouter>{renderRoutes(routes)}</BrowserRouter>, document.getElementById('root'));

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();