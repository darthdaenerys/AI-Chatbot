import React, { useRef } from 'react'
import io from 'socket.io-client';
import { useEffect, useState } from 'react';


export default function Chatbox() {
	const [input, setInput] = useState("");

	const createNewChatElement=()=>{
		let rawhtml = `<div class="message-wrapper"><div class="message left"><p><span id="cursor">|</span></p></div></div>`;
		let chatsElement=document.querySelector('.chats');
		chatsElement.insertAdjacentHTML('beforeend',rawhtml);
	}
	

	const updateChatElement = (word) => {
		let cursor=document.querySelector('#cursor');
		cursor.insertAdjacentHTML('beforebegin',word)
		let chatsElement=document.querySelector('.chats');
		chatsElement.scrollTop=chatsElement.scrollHeight;
	}

	const addUserChatElement=(text)=>{
		let chatsElement=document.querySelector('.chats');
		let rawhtml = `<div class="message-wrapper"><div class="message right"><p>${text}</p></div></div>`;
		chatsElement.insertAdjacentHTML('beforeend', rawhtml);
		chatsElement.scrollTop=chatsElement.scrollHeight;
	}


	
}
