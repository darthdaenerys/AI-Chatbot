import React from 'react'

export default function Navbar(props) {
	return (
		<>
			<nav>
				<i className="fa-solid fa-robot"></i>
				<h2>{props.title}</h2>
			</nav>
		</>
	)
}
